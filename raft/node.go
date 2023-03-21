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

package raft

import (
	"context"
	"errors"

	pb "go.etcd.io/etcd/raft/v3/raftpb"
)

type SnapshotStatus int

const (
	SnapshotFinish  SnapshotStatus = 1
	SnapshotFailure SnapshotStatus = 2
)

var (
	emptyState = pb.HardState{}

	// ErrStopped is returned by methods on Nodes that have been stopped.
	ErrStopped = errors.New("raft: stopped")
)

// SoftState provides state that is useful for logging and debugging.
// The state is volatile and does not need to be persisted to the WAL.
type SoftState struct {
	Lead      uint64 // must use atomic operations to access; keep 64-bit aligned.
	RaftState StateType
}

func (a *SoftState) equal(b *SoftState) bool {
	return a.Lead == b.Lead && a.RaftState == b.RaftState
}

// Ready encapsulates the entries and messages that are ready to read,
// be saved to stable storage, committed or sent to other peers.
// All fields in Ready are read-only.
type Ready struct {
	// 软状态是易变的，包括：当前集群leader、当前节点状态
	*SoftState

	// 硬状态需要被保存，包括：节点当前Term、Vote、Commit
	// raft 节点的当前任期、投票归属和已提交日志索引，
	// 这些信息都需要持久化存储，节点宕机后重启亦可恢复如初；
	pb.HardState

	// 主要用来实现线性读功能
	ReadStates []ReadState

	// 本轮算法层产生的预写日志，此时还未持久化，需要传输到应用层，由应用层完成持久化.
	Entries []pb.Entry

	// 需要被持久化的快照
	Snapshot pb.Snapshot

	// 已经被提交的日志可以应用于数据状态机中
	CommittedEntries []pb.Entry

	// 本轮算法层需要发出的消息，由应用层调用网络通信模块发出.
	Messages []pb.Message

	// MustSync indicates whether the HardState and Entries must be synchronously
	// written to disk or if an asynchronous write is permissible.
	MustSync bool
}

func isHardStateEqual(a, b pb.HardState) bool {
	return a.Term == b.Term && a.Vote == b.Vote && a.Commit == b.Commit
}

// IsEmptyHardState returns true if the given HardState is empty.
func IsEmptyHardState(st pb.HardState) bool {
	return isHardStateEqual(st, emptyState)
}

// IsEmptySnap returns true if the given Snapshot is empty.
func IsEmptySnap(sp pb.Snapshot) bool {
	return sp.Metadata.Index == 0
}

func (rd Ready) containsUpdates() bool {
	return rd.SoftState != nil || !IsEmptyHardState(rd.HardState) ||
		!IsEmptySnap(rd.Snapshot) || len(rd.Entries) > 0 ||
		len(rd.CommittedEntries) > 0 || len(rd.Messages) > 0 || len(rd.ReadStates) != 0
}

// appliedCursor extracts from the Ready the highest index the client has
// applied (once the Ready is confirmed via Advance). If no information is
// contained in the Ready, returns zero.
func (rd Ready) appliedCursor() uint64 {
	if n := len(rd.CommittedEntries); n > 0 {
		return rd.CommittedEntries[n-1].Index
	}
	if index := rd.Snapshot.Metadata.Index; index > 0 {
		return index
	}
	return 0
}

// Node represents a node in a raft cluster.
// Node 是算法层中 raft 节点的抽象，也是应用层与算法层交互的唯一入口，应用层持有Node 作为算法层 raft 节点的引用，
// 通过调用 Node 接口的几个 api，完成与算法层的 channel 通信.
type Node interface {
	// Tick ：传送定时驱动信号，每次调用 Tick 方法的时间间隔是固定的，称为一个 tick，是 raft 节点的最小计时单位，
	// 后续 leader 节点的心跳计时和 leader/candidate 的选举计时也都是以 tick 作为时间单位；
	// 实际上，该方法为应用层向 node.tickc channel 中传入一个信号，而算法层的 goroutine 是处于时刻监听该 channel 状态的，
	// 获取到信号后就会驱动 rawNode 进行定时函数处理；
	// 对于 leader 而言，定时函数中的任务是要向集群中的其他节点广播心跳；对于 follower 和 candidate 而言，定时函数的任务是发起竞选；
	// 此外，对于不同角色，在接收到消息时的处理模式也不尽相同，这部分区别会在角色专属的状态机函数中体现.
	// node实现了Tick方法，该方法主要是用于调用RawNode中的Tick方法，RawNode中的Tick方法会根据节点的不同角色从而调用不同的Tick方法
	Tick()

	// Campaign causes the Node to transition to candidate state and start campaigning to become leader.
	Campaign(ctx context.Context) error

	// Propose proposes that data be appended to the log. Note that proposals can be lost without
	// notice, therefore it is user's job to ensure proposal retries.
	// 应用层调用 Node.Propose 方法，向算法层发起一笔写数据的请求.
	Propose(ctx context.Context, data []byte) error

	// ProposeConfChange 应用层通过该方法提议配置更改
	ProposeConfChange(ctx context.Context, cc pb.ConfChangeI) error

	// Step advances the state machine using the given message. ctx.Err() will be returned, if any.
	// 应用层调用 Node.Step 方法，可以向算法层 goroutine 传送一条自由指定类型的消息，但类型不能是 MsgHup（驱动本节点进行选举）或者
	// MsgBeat（驱动本 leader 节点进行心跳广播），因为这些动作应该是由定时器驱动，而非人为调用.
	Step(ctx context.Context, msg pb.Message) error

	// 应用层调用 Node.Ready 方法返回 node.readyc channel 用于监听，当应用层从 readyc 中读取到 Ready 信号时，说明算法层已经产生了新一轮的处理结果，应用层需要进行响应处理；
	// 当应用层处理完毕后，需要调用 Node.Advance 方法，通过向 node.advancec channel 中发送信号的方式，示意应用层已处理完毕，算法层可以进入下一轮调度循环.
	// 因此，Node.Ready 和 Node.Advance 方法是成对使用的，当两个方法各被调用一次，意味着应用层与算法层之间完成了一轮交互，紧接着会开启下一轮，周而复始，循环往复.

	Ready() <-chan Ready

	Advance()

	// ApplyConfChange 应用层调用该方法，通过 node.propcc 向算法层发送一则消息类型为 MsgProp、日志类型为 EntryConfChange 的消息，
	// 推动 raft 节点进入配置变更流程.
	ApplyConfChange(cc pb.ConfChangeI) *pb.ConfState

	// TransferLeadership attempts to transfer leadership to the given transferee.
	TransferLeadership(ctx context.Context, lead, transferee uint64)

	// ReadIndex 应用层调用 node.ReadIndex 方法向算法层发起读数据请求.
	// 实际上会往 node.recvc channel 中传入一条类型为 MsgReadIndex 的消息.
	ReadIndex(ctx context.Context, rctx []byte) error

	// Status returns the current status of the raft state machine.
	Status() Status
	// ReportUnreachable reports the given node is not reachable for the last send.
	// 告诉算法层节点不可达
	ReportUnreachable(id uint64)

	// ReportSnapshot 报告某个节点的快照同步是否成功
	ReportSnapshot(id uint64, status SnapshotStatus)
	// Stop performs any necessary termination of the Node.
	// 关闭节点
	Stop()
}

type Peer struct {
	ID      uint64
	Context []byte
}

// StartNode returns a new Node given configuration and a list of raft peers.
// It appends a ConfChangeAddNode entry for each given peer to the initial log.
//
// Peers must not be zero length; call RestartNode in that case.
func StartNode(c *Config, peers []Peer) Node {
	if len(peers) == 0 {
		panic("no peers given; use RestartNode instead")
	}
	rn, err := NewRawNode(c)
	if err != nil {
		panic(err)
	}
	rn.Bootstrap(peers)

	n := newNode(rn)

	go n.run()
	return &n
}

// RestartNode is similar to StartNode but does not take a list of peers.
// The current membership of the cluster will be restored from the Storage.
// If the caller has an existing state machine, pass in the last log index that
// has been applied to it; otherwise use zero.
func RestartNode(c *Config) Node {
	rn, err := NewRawNode(c)
	if err != nil {
		panic(err)
	}
	n := newNode(rn)
	go n.run()
	return &n
}

type msgWithResult struct {
	m      pb.Message
	result chan error
}

// node is the canonical implementation of the Node interface
type node struct {
	propc      chan msgWithResult
	recvc      chan pb.Message
	confc      chan pb.ConfChangeV2
	confstatec chan pb.ConfState
	readyc     chan Ready
	advancec   chan struct{}
	tickc      chan struct{}
	done       chan struct{}
	stop       chan struct{}
	status     chan chan Status

	rn *RawNode
}

func newNode(rn *RawNode) node {
	return node{
		propc:      make(chan msgWithResult),
		recvc:      make(chan pb.Message),
		confc:      make(chan pb.ConfChangeV2),
		confstatec: make(chan pb.ConfState),
		readyc:     make(chan Ready),
		advancec:   make(chan struct{}),
		// make tickc a buffered chan, so raft node can buffer some ticks when the node
		// is busy processing raft messages. Raft node will resume process buffered
		// ticks when it becomes idle.
		tickc:  make(chan struct{}, 128),
		done:   make(chan struct{}),
		stop:   make(chan struct{}),
		status: make(chan chan Status),
		rn:     rn,
	}
}

func (n *node) Stop() {
	select {
	case n.stop <- struct{}{}:
		// Not already stopped, so trigger it
	case <-n.done:
		// Node has already been stopped - no need to do anything
		return
	}
	// Block until the stop has been acknowledged by run()
	<-n.done
}

// 启动算法层 raft 节点 goroutine，未来正是这个 goroutine 持续与应用层进行通信交互
func (n *node) run() {
	var propc chan msgWithResult
	var readyc chan Ready
	var advancec chan struct{}
	var rd Ready
	r := n.rn.raft
	lead := None
	for {
		if advancec != nil {
			// advance channel不为空，说明还在等应用调用Advance接口通知已经处理完毕了本次的ready数据
			readyc = nil
		} else if n.rn.HasReady() {
			// Populate a Ready. Note that this Ready is not guaranteed to
			// actually be handled. We will arm readyc, but there's no guarantee
			// that we will actually send on it. It's possible that we will
			// service another channel instead, loop around, and then populate
			// the Ready again. We could instead force the previous Ready to be
			// handled first, but it's generally good to emit larger Readys plus
			// it simplifies testing (by emitting less frequently and more
			// predictably).
			rd = n.rn.readyWithoutAccept()
			readyc = n.readyc
		}

		if lead != r.lead {
			if r.hasLeader() {
				if lead == None {
					r.logger.Infof("raft.node: %x elected leader %x at term %d", r.id, r.lead, r.Term)
				} else {
					r.logger.Infof("raft.node: %x changed leader from %x to %x at term %d", r.id, lead, r.lead, r.Term)
				}
				propc = n.propc
			} else {
				r.logger.Infof("raft.node: %x lost leader %x at term %d", r.id, lead, r.Term)
				propc = nil
			}
			lead = r.lead
		}

		select {
		// TODO: maybe buffer the config propose if there exists one (the way
		// described in raft dissertation)
		// Currently it is dropped in Step silently.
		case pm := <-propc:
			// 接收来自应用层的的消息
			m := pm.m
			m.From = r.id
			err := r.Step(m)
			if pm.result != nil {
				pm.result <- err
				close(pm.result)
			}
		case m := <-n.recvc:
			// 处理其他节点发送过来的提交值
			// filter out response message from unknown From.
			if pr := r.prs.Progress[m.From]; pr != nil || !IsResponseMsg(m.Type) {
				r.Step(m)
			}
		case cc := <-n.confc:
			_, okBefore := r.prs.Progress[r.id]
			cs := r.applyConfChange(cc)
			// If the node was removed, block incoming proposals. Note that we
			// only do this if the node was in the config before. Nodes may be
			// a member of the group without knowing this (when they're catching
			// up on the log and don't have the latest config) and we don't want
			// to block the proposal channel in that case.
			//
			// NB: propc is reset when the leader changes, which, if we learn
			// about it, sort of implies that we got readded, maybe? This isn't
			// very sound and likely has bugs.
			if _, okAfter := r.prs.Progress[r.id]; okBefore && !okAfter {
				var found bool
			outer:
				for _, sl := range [][]uint64{cs.Voters, cs.VotersOutgoing} {
					for _, id := range sl {
						if id == r.id {
							found = true
							break outer
						}
					}
				}
				if !found {
					propc = nil
				}
			}
			select {
			case n.confstatec <- cs:
			case <-n.done:
			}
		case <-n.tickc:
			//接收到应用层的定时 Tick 调用时，会根据 raft 节点的角色，使用其对应的 tick 函数进行处理：
			n.rn.Tick()
		case readyc <- rd:
			n.rn.acceptReady(rd)
			// 修改advance channel不为空，等待接收advance消息
			advancec = n.advancec
		case <-advancec:
			n.rn.Advance(rd)
			rd = Ready{}
			advancec = nil
		case c := <-n.status:
			c <- getStatus(r)
		case <-n.stop:
			close(n.done)
			return
		}
	}
}

// Tick increments the internal logical clock for this Node. Election timeouts
// and heartbeat timeouts are in units of ticks.
func (n *node) Tick() {
	select {
	case n.tickc <- struct{}{}:
	case <-n.done:
	default:
		n.rn.raft.logger.Warningf("%x A tick missed to fire. Node blocks too long!", n.rn.raft.id)
	}
}

func (n *node) Campaign(ctx context.Context) error { return n.step(ctx, pb.Message{Type: pb.MsgHup}) }

func (n *node) Propose(ctx context.Context, data []byte) error {
	return n.stepWait(ctx, pb.Message{Type: pb.MsgProp, Entries: []pb.Entry{{Data: data}}})
}

func (n *node) Step(ctx context.Context, m pb.Message) error {
	// ignore unexpected local messages receiving over network
	if IsLocalMsg(m.Type) {
		// TODO: return an error?
		return nil
	}
	return n.step(ctx, m)
}

func confChangeToMsg(c pb.ConfChangeI) (pb.Message, error) {
	typ, data, err := pb.MarshalConfChange(c)
	if err != nil {
		return pb.Message{}, err
	}
	return pb.Message{Type: pb.MsgProp, Entries: []pb.Entry{{Type: typ, Data: data}}}, nil
}

func (n *node) ProposeConfChange(ctx context.Context, cc pb.ConfChangeI) error {
	msg, err := confChangeToMsg(cc)
	if err != nil {
		return err
	}
	return n.Step(ctx, msg)
}

func (n *node) step(ctx context.Context, m pb.Message) error {
	return n.stepWithWaitOption(ctx, m, false)
}

func (n *node) stepWait(ctx context.Context, m pb.Message) error {
	return n.stepWithWaitOption(ctx, m, true)
}

// Step advances the state machine using msgs. The ctx.Err() will be returned,
// if any.
func (n *node) stepWithWaitOption(ctx context.Context, m pb.Message, wait bool) error {
	if m.Type != pb.MsgProp {
		select {
		case n.recvc <- m:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		case <-n.done:
			return ErrStopped
		}
	}
	ch := n.propc
	pm := msgWithResult{m: m}
	if wait {
		pm.result = make(chan error, 1)
	}
	select {
	case ch <- pm:
		if !wait {
			return nil
		}
	case <-ctx.Done():
		return ctx.Err()
	case <-n.done:
		return ErrStopped
	}
	select {
	case err := <-pm.result:
		if err != nil {
			return err
		}
	case <-ctx.Done():
		return ctx.Err()
	case <-n.done:
		return ErrStopped
	}
	return nil
}

func (n *node) Ready() <-chan Ready { return n.readyc }

func (n *node) Advance() {
	select {
	case n.advancec <- struct{}{}:
	case <-n.done:
	}
}

func (n *node) ApplyConfChange(cc pb.ConfChangeI) *pb.ConfState {
	var cs pb.ConfState
	select {
	case n.confc <- cc.AsV2():
	case <-n.done:
	}
	select {
	case cs = <-n.confstatec:
	case <-n.done:
	}
	return &cs
}

func (n *node) Status() Status {
	c := make(chan Status)
	select {
	case n.status <- c:
		return <-c
	case <-n.done:
		return Status{}
	}
}

func (n *node) ReportUnreachable(id uint64) {
	select {
	case n.recvc <- pb.Message{Type: pb.MsgUnreachable, From: id}:
	case <-n.done:
	}
}

func (n *node) ReportSnapshot(id uint64, status SnapshotStatus) {
	rej := status == SnapshotFailure

	select {
	case n.recvc <- pb.Message{Type: pb.MsgSnapStatus, From: id, Reject: rej}:
	case <-n.done:
	}
}

func (n *node) TransferLeadership(ctx context.Context, lead, transferee uint64) {
	select {
	// manually set 'from' and 'to', so that leader can voluntarily transfers its leadership
	case n.recvc <- pb.Message{Type: pb.MsgTransferLeader, From: transferee, To: lead}:
	case <-n.done:
	case <-ctx.Done():
	}
}

func (n *node) ReadIndex(ctx context.Context, rctx []byte) error {
	return n.step(ctx, pb.Message{Type: pb.MsgReadIndex, Entries: []pb.Entry{{Data: rctx}}})
}

func newReady(r *raft, prevSoftSt *SoftState, prevHardSt pb.HardState) Ready {
	rd := Ready{
		Entries:          r.raftLog.unstableEntries(),
		CommittedEntries: r.raftLog.nextEnts(),
		Messages:         r.msgs,
	}
	if softSt := r.softState(); !softSt.equal(prevSoftSt) {
		rd.SoftState = softSt
	}
	if hardSt := r.hardState(); !isHardStateEqual(hardSt, prevHardSt) {
		rd.HardState = hardSt
	}
	if r.raftLog.unstable.snapshot != nil {
		rd.Snapshot = *r.raftLog.unstable.snapshot
	}
	if len(r.readStates) != 0 {
		rd.ReadStates = r.readStates
	}
	rd.MustSync = MustSync(r.hardState(), prevHardSt, len(rd.Entries))
	return rd
}

// MustSync returns true if the hard state and count of Raft entries indicate
// that a synchronous write to persistent storage is required.
func MustSync(st, prevst pb.HardState, entsnum int) bool {
	// Persistent state on all servers:
	// (Updated on stable storage before responding to RPCs)
	// currentTerm
	// votedFor
	// log entries[]
	return entsnum != 0 || st.Vote != prevst.Vote || st.Term != prevst.Term
}
