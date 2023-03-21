// Copyright 2015 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"bytes"
	"encoding/gob"
	"encoding/json"
	"log"
	"sync"

	"go.etcd.io/etcd/raft/v3/raftpb"
	"go.etcd.io/etcd/server/v3/etcdserver/api/snap"
)

// a key-value store backed by raft
// kvstore 是一个乞丐版的键值对存储模块，数据只是通过一个map结构简单的存储在内存中，
// 可以通过snapshotter持久化，我们可以自定义一个存储代替该存储
type kvstore struct {
	proposeC    chan<- string // channel for proposing updates
	mu          sync.RWMutex
	kvStore     map[string]string // current committed key-value pairs
	snapshotter *snap.Snapshotter
}

type kv struct {
	Key string
	Val string
}

func newKVStore(snapshotter *snap.Snapshotter, proposeC chan<- string, commitC <-chan *commit, errorC <-chan error) *kvstore {
	s := &kvstore{proposeC: proposeC, kvStore: make(map[string]string), snapshotter: snapshotter}
	snapshot, err := s.loadSnapshot()
	if err != nil {
		log.Panic(err)
	}
	if snapshot != nil {
		log.Printf("loading snapshot at term %d and index %d", snapshot.Metadata.Term, snapshot.Metadata.Index)
		if err := s.recoverFromSnapshot(snapshot.Data); err != nil {
			log.Panic(err)
		}
	}
	// read commits from raft into kvStore map until error
	go s.readCommits(commitC, errorC)
	return s
}

// 根据key返回val
func (s *kvstore) Lookup(key string) (string, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	v, ok := s.kvStore[key]
	return v, ok
}

// 提议一个key-val
func (s *kvstore) Propose(k string, v string) {
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(kv{k, v}); err != nil {
		log.Fatal(err)
	}
	s.proposeC <- buf.String()
}

// 读取commitC管道，该管道是可以提交的Entry，需要持久化到内存中
func (s *kvstore) readCommits(commitC <-chan *commit, errorC <-chan error) {
	for commit := range commitC {
		if commit == nil {
			// 当commit == nil时代表此时从从快照中恢复内存数据
			snapshot, err := s.loadSnapshot()
			if err != nil {
				log.Panic(err)
			}
			if snapshot != nil {
				log.Printf("loading snapshot at term %d and index %d", snapshot.Metadata.Term, snapshot.Metadata.Index)
				if err := s.recoverFromSnapshot(snapshot.Data); err != nil {
					log.Panic(err)
				}
			}
			continue
		}

		for _, data := range commit.data {
			var dataKv kv
			dec := gob.NewDecoder(bytes.NewBufferString(data))
			if err := dec.Decode(&dataKv); err != nil {
				log.Fatalf("raftexample: could not decode message (%v)", err)
			}
			s.mu.Lock()
			s.kvStore[dataKv.Key] = dataKv.Val
			s.mu.Unlock()
		}
		//通过chanel传达消息，表示该Entry已经持久化
		close(commit.applyDoneC)
	}
	if err, ok := <-errorC; ok {
		log.Fatal(err)
	}
}

// 获取当前内存快照
func (s *kvstore) getSnapshot() ([]byte, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return json.Marshal(s.kvStore)
}

// 加载快照
func (s *kvstore) loadSnapshot() (*raftpb.Snapshot, error) {
	snapshot, err := s.snapshotter.Load()
	if err == snap.ErrNoSnapshot {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return snapshot, nil
}

// 根据快照恢复内存数据
func (s *kvstore) recoverFromSnapshot(snapshot []byte) error {
	var store map[string]string
	if err := json.Unmarshal(snapshot, &store); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.kvStore = store
	return nil
}
