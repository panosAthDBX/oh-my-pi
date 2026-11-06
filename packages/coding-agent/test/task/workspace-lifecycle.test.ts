import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "bun:test";
import * as fsSync from "node:fs";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { type } from "@oh-my-pi/omptype";
import { AgentLifecycleManager } from "@oh-my-pi/pi-coding-agent/registry/agent-lifecycle";
import { AgentRegistry } from "@oh-my-pi/pi-coding-agent/registry/agent-registry";
import type { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import * as executorModule from "@oh-my-pi/pi-coding-agent/task/executor";
import {
	nativeWorkspace,
	prepareIsolationContext,
	runIsolatedSubprocess,
} from "@oh-my-pi/pi-coding-agent/task/isolation-runner";
import { getTaskSchema } from "@oh-my-pi/pi-coding-agent/task/types";
import {
	getWorkspaceLifecycleHost,
	getWorkspaceLifecycleService,
	WORKSPACE_LEASE_FILE,
	type WorkspaceLease,
	WorkspaceLifecycleService,
	type WorkspaceReleaseGate,
} from "@oh-my-pi/pi-coding-agent/task/workspace-lifecycle";
import * as worktreeModule from "@oh-my-pi/pi-coding-agent/task/worktree";
import * as natives from "@oh-my-pi/pi-natives";
import { postmortem, setWorktreesDir } from "@oh-my-pi/pi-utils";
import type { SingleResult } from "@oh-my-pi/pi-tui/tools/task";

const host = getWorkspaceLifecycleHost();
let root: string;
let repo: string;
let wtDir: string;
let firstCommit: string;
let secondCommit: string;
const disposeGates: Array<() => void> = [];

async function git(cwd: string, ...args: string[]): Promise<string> {
	const proc = Bun.spawn(["git", ...args], { cwd, stdout: "pipe", stderr: "pipe" });
	const [out, err, code] = await Promise.all([
		new Response(proc.stdout).text(),
		new Response(proc.stderr).text(),
		proc.exited,
	]);
	if (code !== 0) throw new Error(`git ${args.join(" ")}: ${err}`);
	return out.trim();
}

beforeAll(async () => {
	root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), "omp-wl-companion-")));
	repo = path.join(root, "repo");
	wtDir = path.join(root, "wt");
	await fs.mkdir(repo);
	setWorktreesDir(wtDir);
	process.env.OMP_WORKTREE_DIR = wtDir;
	await git(repo, "init", "-q", "-b", "main");
	await git(repo, "config", "user.email", "t@example.com");
	await git(repo, "config", "user.name", "t");
	await git(repo, "config", "core.autocrlf", "false");
	await Bun.write(path.join(repo, ".gitignore"), "ignored.log\n");
	await Bun.write(path.join(repo, "a.txt"), "one\n");
	await git(repo, "add", ".");
	await git(repo, "commit", "-qm", "one");
	firstCommit = await git(repo, "rev-parse", "HEAD");
	await Bun.write(path.join(repo, "a.txt"), "two\n");
	await git(repo, "commit", "-qam", "two");
	secondCommit = await git(repo, "rev-parse", "HEAD");
	// Root dirty state a fresh successor must NOT inherit.
	await Bun.write(path.join(repo, "a.txt"), "dirty\n");
	await Bun.write(path.join(repo, "untracked.txt"), "stray\n");
	await Bun.write(path.join(repo, "ignored.log"), "kept\n");
});

afterEach(() => {
	for (const dispose of disposeGates.splice(0)) dispose();
	vi.restoreAllMocks();
});

afterAll(async () => {
	setWorktreesDir(undefined);
	delete process.env.OMP_WORKTREE_DIR;
	await fs.rm(root, { recursive: true, force: true });
});

function gate(fn: WorkspaceReleaseGate): void {
	disposeGates.push(host.registerReleaseGate(fn));
}

function lease(id: string): WorkspaceLease {
	const found = host.list().find(l => l.agentId === id);
	if (!found) throw new Error(`no lease for ${id}`);
	return found;
}

async function exists(p: string): Promise<boolean> {
	return fs.stat(p).then(
		() => true,
		() => false,
	);
}

interface SpawnOptions {
	keepAlive?: boolean;
	baseline?: string;
	/** Runs inside the fake child while its registry status is `running`. */
	whileRunning?: (worktree: string) => Promise<void>;
	onSubprocessResult?: () => void;
	/** The child session's dispose (the first step of a native release, before capture and gates). */
	sessionDispose?: () => Promise<void>;
}

/** Real isolation runner + lifecycle; only the model subprocess is faked. */
async function spawn(id: string, opts: SpawnOptions = {}): Promise<SingleResult> {
	const keepAlive = opts.keepAlive ?? true;
	const artifactsDir = path.join(root, "artifacts");
	await fs.mkdir(artifactsDir, { recursive: true });
	const session = {
		prepareForHeadlessAdvisorDrain: () => {},
		waitForAdvisorCatchup: async () => true,
		dispose: opts.sessionDispose ?? (async () => {}),
	} as unknown as AgentSession;
	const base: SingleResult = {
		index: 0,
		id,
		agent: "task",
		agentSource: "bundled",
		task: "work",
		assignment: "work",
		exitCode: 0,
		output: "done",
		stderr: "",
		truncated: false,
		durationMs: 1,
		tokens: 0,
		requests: 0,
	};
	vi.spyOn(executorModule, "runSubprocess").mockImplementation(async options => {
		AgentRegistry.global().register({
			id: options.id,
			displayName: options.id,
			kind: "sub",
			session,
			sessionFile: path.join(root, `${options.id}.jsonl`),
			status: "running",
		});
		await opts.whileRunning?.(options.worktree ?? "");
		await Bun.write(path.join(options.worktree ?? "", "work.txt"), `${options.id}\n`);
		await executorModule.finalizeSubagentLifecycle({
			id: options.id,
			session,
			aborted: false,
			keepAlive,
			isolated: true,
			agentIdleTtlMs: 0,
			reviveSession: async () => session,
			onRelease: options.onRelease,
		});
		return base;
	});
	return runIsolatedSubprocess({
		baseOptions: {
			cwd: repo,
			agent: { name: "task", description: "Task agent", systemPrompt: "test", source: "bundled" },
			task: "work",
			index: 0,
			id,
			keepAlive,
		},
		context: await prepareIsolationContext(repo),
		openWorkspace: nativeWorkspace(undefined),
		agentId: id,
		mergeMode: "patch",
		artifactsDir,
		baseline: opts.baseline,
		onSubprocessResult: opts.onSubprocessResult,
		buildFailureResult: error => ({ ...base, exitCode: 1, error: String(error) }),
	});
}

describe("workspace lifecycle host companion", () => {
	it("exposes apiVersion 1 with one identity", () => {
		expect(host.apiVersion).toBe(1);
		expect(typeof host.hostVersion).toBe("string");
		expect(getWorkspaceLifecycleHost()).toBe(host);
		expect(Object.isFrozen(host)).toBe(true);
		expect("register" in host).toBe(false);
	});

	it("materialises an explicit baseline exactly, without root dirty edits", async () => {
		let seen: { head: string; a: string; untracked: boolean; ignored: boolean } | undefined;
		await spawn("BaselineExact", {
			baseline: firstCommit.slice(0, 10),
			whileRunning: async wt => {
				seen = {
					head: await git(wt, "rev-parse", "HEAD"),
					a: await Bun.file(path.join(wt, "a.txt")).text(),
					untracked: await exists(path.join(wt, "untracked.txt")),
					ignored: await exists(path.join(wt, "ignored.log")),
				};
			},
		});
		expect(seen).toEqual({ head: firstCommit, a: "one\n", untracked: false, ignored: true });
		const l = lease("BaselineExact");
		expect(l.baseline).toEqual({
			commit: firstCommit,
			requested: firstCommit.slice(0, 10),
			source: "explicit-commit",
			nested: "source-working-copy",
		});
		const persisted = await Bun.file(path.join(l.baseDir, WORKSPACE_LEASE_FILE)).json();
		expect(persisted.generation).toBe(l.generation);
		expect(persisted.baseline.commit).toBe(firstCommit);
		expect(typeof persisted.sessionFile).toBe("string");
		// The patch captures only the child's work relative to the explicit commit.
		expect(await Bun.file(l.capture?.patchPath ?? "").text()).toContain("+BaselineExact");
		expect((await host.release("BaselineExact", { generation: l.generation, reason: "test" })).status).toBe(
			"released",
		);
	});

	it("removes root untracked nested repos from an explicit baseline", async () => {
		const nested = path.join(repo, "experiments");
		await fs.mkdir(nested);
		await git(nested, "init", "-q");
		await Bun.write(path.join(nested, "e.txt"), "e\n");
		let leaked: boolean | undefined;
		try {
			await spawn("BaselineNested", {
				baseline: firstCommit,
				whileRunning: async wt => {
					leaked = await exists(path.join(wt, "experiments"));
				},
			});
		} finally {
			await fs.rm(nested, { recursive: true, force: true });
		}
		expect(leaked).toBe(false);
		await host.release("BaselineNested", { generation: lease("BaselineNested").generation, reason: "test" });
	});

	it("records working-tree baselines by default", async () => {
		await spawn("BaselineDefault");
		const l = lease("BaselineDefault");
		expect(l.baseline.source).toBe("working-tree");
		expect(l.baseline.commit).toBe(secondCommit);
		await host.release("BaselineDefault", { generation: l.generation, reason: "test" });
	});

	it("refuses an unresolvable baseline before creating any workspace", async () => {
		const before = await fs.readdir(wtDir).catch(() => [] as string[]);
		const result = await spawn("BaselineBad", { baseline: "no-such-ref" });
		expect(result.exitCode).toBe(1);
		expect(result.error).toContain('Task baseline "no-such-ref"');
		expect(await fs.readdir(wtDir).catch(() => [] as string[])).toEqual(before);
		expect(host.list().some(l => l.agentId === "BaselineBad")).toBe(false);
	});

	it("refuses explicit release while the runner is still capturing", async () => {
		let during: Promise<{ status: string; reason?: string }> | undefined;
		await spawn("Capturing", {
			onSubprocessResult: () => {
				during = host.release("Capturing", { generation: lease("Capturing").generation, reason: "x" });
			},
		});
		expect(await during).toMatchObject({ status: "refused", reason: "active-owner: capture in progress" });
		const l = lease("Capturing");
		expect(await Bun.file(l.capture?.patchPath ?? "").text()).toContain("+Capturing");
		expect((await host.release("Capturing", { generation: l.generation, reason: "x" })).status).toBe("released");
	});

	it("holds a superseded lease's workspace instead of deleting it ungated", async () => {
		await spawn("Superseded");
		const old = lease("Superseded");
		const service = getWorkspaceLifecycleService();
		const otherBase = path.join(root, "superseding");
		await fs.mkdir(otherBase, { recursive: true });
		const next = service.register(
			{ ...old, baseDir: otherBase, workspaceDir: path.join(otherBase, "m") },
			async () => {},
		);
		// Native idle release of the earlier run: its gate lookup no longer finds a current entry.
		await AgentLifecycleManager.global().release("Superseded");
		const parent = path.dirname(old.baseDir);
		const retained = (await fs.readdir(parent)).find(n => n.startsWith(`${path.basename(old.baseDir)}.retained-`));
		expect(retained).toBeString();
		expect(await Bun.file(path.join(parent, retained ?? "", "m", "work.txt")).text()).toBe("Superseded\n");
		service.update("Superseded", next, { state: "released" });
		await fs.rm(path.join(parent, retained ?? ""), { recursive: true, force: true });
	});

	for (const sameSlot of [false, true]) {
		it(`preserves source and successor when superseded inside a gate (same slot: ${sameSlot})`, async () => {
			const id = sameSlot ? "GateReusedSlot" : "GateReplacedLease";
			await spawn(id);
			const old = lease(id);
			const entered = Promise.withResolvers<void>();
			const resume = Promise.withResolvers<void>();
			gate(async () => {
				entered.resolve();
				await resume.promise;
				return { ok: true };
			});
			const releasing = host.release(id, { generation: old.generation, reason: "retire old" });
			await entered.promise;
			const baseDir = sameSlot ? old.baseDir : path.join(root, `${id}-successor`);
			const workspaceDir = path.join(baseDir, "m");
			await fs.mkdir(workspaceDir, { recursive: true });
			await Bun.write(path.join(workspaceDir, "successor.txt"), "successor source\n");
			const service = getWorkspaceLifecycleService();
			const next = service.register({ ...old, baseDir, workspaceDir }, async () => {});
			await service.holdAtShutdown(id, "successor retained");
			const sidecar = path.join(baseDir, WORKSPACE_LEASE_FILE);
			const before = await Bun.file(sidecar).text();
			try {
				resume.resolve();
				const result = await releasing;
				expect(result.status).toBe("held");
				expect(await Bun.file(path.join(workspaceDir, "successor.txt")).text()).toBe("successor source\n");
				expect(await Bun.file(sidecar).text()).toBe(before);
				expect(lease(id).generation).toBe(next);
				if (sameSlot) {
					expect(result.reason).toContain("left in place");
					expect(await Bun.file(path.join(old.workspaceDir, "work.txt")).text()).toBe(`${id}\n`);
				} else {
					const names = await fs.readdir(path.dirname(old.baseDir));
					const retained = names.find(name => name.startsWith(`${path.basename(old.baseDir)}.retained-`));
					expect(retained).toBeString();
					expect(
						await Bun.file(path.join(path.dirname(old.baseDir), retained ?? "", "m", "work.txt")).text(),
					).toBe(`${id}\n`);
				}
			} finally {
				resume.resolve();
				await releasing;
			}
		});
	}

	it("holds the workspace when a gate holds", async () => {
		gate(async () => ({ ok: false, hold: "preservation unverified" }));
		await spawn("GateHold");
		const before = lease("GateHold");
		const result = await host.release("GateHold", { generation: before.generation, reason: "retire" });
		expect(result.status).toBe("held");
		expect(result.reason).toBe("preservation unverified");
		const retained = result.lease?.retainedDir ?? "";
		expect(retained).toContain(".retained-");
		expect(await Bun.file(path.join(retained, "work.txt")).text()).toBe("GateHold\n");
		const persisted = await Bun.file(path.join(path.dirname(retained), WORKSPACE_LEASE_FILE)).json();
		expect(persisted).toMatchObject({ state: "held", holdReason: "preservation unverified", retainedDir: retained });
		expect((await host.release("GateHold", { generation: before.generation, reason: "again" })).status).toBe("held");
		await fs.rm(path.dirname(retained), { recursive: true, force: true });
	});

	it("treats a throwing gate as a hold", async () => {
		gate(async () => {
			throw new Error("boom");
		});
		await spawn("GateThrow");
		const result = await host.release("GateThrow", { generation: lease("GateThrow").generation, reason: "x" });
		expect(result.status).toBe("held");
		expect(result.reason).toContain("boom");
		await fs.rm(path.dirname(result.lease?.retainedDir ?? ""), { recursive: true, force: true });
	});

	it("releases and removes after passing gates; repeats and races agree", async () => {
		const seen: Array<{ trigger: string; reason: string; state: string; patch?: string }> = [];
		gate(async (l, ctx) => {
			seen.push({ ...ctx, state: l.state, patch: l.capture?.patchPath });
			return { ok: true };
		});
		await spawn("GateOk");
		const l = lease("GateOk");
		const [a, b] = await Promise.all([
			host.release("GateOk", { generation: l.generation, reason: "integrated" }),
			host.release("GateOk", { generation: l.generation, reason: "integrated" }),
		]);
		expect(a.status).toBe("released");
		expect(b.status).toBe("released");
		expect(seen).toHaveLength(1);
		expect(seen[0]).toMatchObject({ trigger: "explicit", reason: "integrated", state: "releasing" });
		expect(seen[0]?.patch).toBeString();
		expect(await exists(l.baseDir)).toBe(false);
		expect(AgentRegistry.global().get("GateOk")?.status).toBe("aborted");
		const again = await host.release("GateOk", { generation: l.generation, reason: "integrated" });
		expect(again.status).toBe("already-released");
	});

	it("refuses unknown agents, stale generations and running owners", async () => {
		expect((await host.release("Nobody", { generation: "1", reason: "x" })).reason).toBe("unknown-agent");
		let running: string | undefined;
		await spawn("Owner", {
			whileRunning: async () => {
				const r = await host.release("Owner", { generation: lease("Owner").generation, reason: "x" });
				running = r.reason;
				expect(lease("Owner").state).toBe("active");
			},
		});
		expect(running).toBe("active-owner");
		expect(lease("Owner").state).toBe("idle");
		expect((await host.release("Owner", { generation: "stale", reason: "x" })).reason).toBe("stale-generation");
		const done = await host.release("Owner", { generation: lease("Owner").generation, reason: "x" });
		expect(done.status).toBe("released");
	});

	it("gates one-shot lifecycle teardown and keeps a held workspace", async () => {
		const triggers: string[] = [];
		gate(async (_l, ctx) => {
			triggers.push(ctx.trigger);
			return { ok: false, hold: "pinned lane" };
		});
		await spawn("OneShot", { keepAlive: false });
		const l = lease("OneShot");
		expect(triggers).toEqual(["lifecycle"]);
		expect(l.state).toBe("held");
		expect(l.holdReason).toBe("pinned lane");
		expect(await exists(path.join(l.retainedDir ?? "", "work.txt"))).toBe(true);
		await fs.rm(path.dirname(l.retainedDir ?? ""), { recursive: true, force: true });
	});

	it("holds instead of deleting when native teardown fails", async () => {
		const realEnsure = worktreeModule.ensureIsolation;
		vi.spyOn(worktreeModule, "ensureIsolation").mockImplementation(async (...args) => ({
			...(await realEnsure(...args)),
			backend: natives.IsoBackendKind.Btrfs,
		}));
		const stop = vi.spyOn(natives, "isoStop").mockRejectedValue(new Error("umount busy"));
		await spawn("NativeStop");
		const result = await host.release("NativeStop", { generation: lease("NativeStop").generation, reason: "x" });
		expect(stop).toHaveBeenCalled();
		expect(result.status).toBe("held");
		expect(result.reason).toContain("umount busy");
		expect(await exists(path.join(result.lease?.retainedDir ?? "", "work.txt"))).toBe(true);
		stop.mockRestore();
		await fs.rm(path.dirname(result.lease?.retainedDir ?? ""), { recursive: true, force: true });
	});

	it("cleanupIsolation leaves a native-teardown workspace in place when stop fails", async () => {
		const base = path.join(root, "native-unit");
		await fs.mkdir(path.join(base, "m"), { recursive: true });
		vi.spyOn(natives, "isoStop").mockRejectedValue(new Error("busy"));
		await expect(
			worktreeModule.cleanupIsolation({
				mergedDir: path.join(base, "m"),
				backend: natives.IsoBackendKind.Overlayfs,
				fellBack: false,
				fallbackReason: null,
			}),
		).rejects.toThrow("busy");
		expect(await exists(base)).toBe(true);
		await worktreeModule.cleanupIsolation({
			mergedDir: path.join(base, "m"),
			backend: natives.IsoBackendKind.Rcopy,
			fellBack: false,
			fallbackReason: null,
		});
		expect(await exists(base)).toBe(false);
	});
});

/** Resolves with the lease once the runner retains its workspace (its `update` carrying `retainedDir`). */
function retainedLease(id: string): Promise<WorkspaceLease> {
	const service = getWorkspaceLifecycleService();
	const update = service.update.bind(service);
	const { promise, resolve } = Promise.withResolvers<WorkspaceLease>();
	vi.spyOn(service, "update").mockImplementation((agentId, generation, patch) => {
		update(agentId, generation, patch);
		if (agentId === id && patch.retainedDir) resolve(lease(id));
	});
	return promise;
}

const SHUTDOWN_HELD = { state: "held", holdReason: "shutdown deadline: release incomplete" };

describe("workspace lifecycle at shutdown", () => {
	it("waits past the agent grace for a slow gated release, then removes", async () => {
		// Real time on purpose: dispose bounds releases with AbortSignal.timeout, which fake timers do not drive.
		gate(async () => {
			await Bun.sleep(6000);
			return { ok: true };
		});
		await spawn("SlowGate");
		const before = lease("SlowGate");
		await AgentLifecycleManager.global().dispose();
		expect(lease("SlowGate").state).toBe("released");
		expect(await exists(before.baseDir)).toBe(false);
	}, 20_000);

	it("persists the lease as held when the gate outlasts the bound, and never removes after", async () => {
		const { promise: gateDone, resolve: finishGate } = Promise.withResolvers<void>();
		gate(async () => {
			await gateDone;
			return { ok: true };
		});
		await spawn("StuckGate");
		const before = lease("StuckGate");
		await AgentLifecycleManager.global().dispose(Date.now(), Date.now() + 300);
		expect(lease("StuckGate")).toMatchObject(SHUTDOWN_HELD);
		expect(await Bun.file(path.join(before.baseDir, WORKSPACE_LEASE_FILE)).json()).toMatchObject(SHUTDOWN_HELD);
		expect(await Bun.file(path.join(before.workspaceDir, "work.txt")).text()).toBe("StuckGate\n");
		// The late gate pass must keep the workspace (retained), not delete it.
		const retained = retainedLease("StuckGate");
		finishGate();
		const after = await retained;
		expect(after).toMatchObject(SHUTDOWN_HELD);
		expect(await Bun.file(path.join(after.retainedDir ?? "", "work.txt")).text()).toBe("StuckGate\n");
		await fs.rm(path.dirname(after.retainedDir ?? ""), { recursive: true, force: true });
	});

	it("a hold that lands before the gates start is never overwritten, and the workspace is retained", async () => {
		const gateCalls: string[] = [];
		gate(async l => {
			gateCalls.push(l.agentId);
			return { ok: true };
		});
		const { promise: stall, resolve: unstall } = Promise.withResolvers<void>();
		await spawn("PreGate", { sessionDispose: () => stall });
		const before = lease("PreGate");
		await AgentLifecycleManager.global().dispose(Date.now(), Date.now() + 300);
		expect(gateCalls).toEqual([]);
		expect(await Bun.file(path.join(before.baseDir, WORKSPACE_LEASE_FILE)).json()).toMatchObject(SHUTDOWN_HELD);
		const retained = retainedLease("PreGate");
		unstall();
		const after = await retained;
		expect(after).toMatchObject(SHUTDOWN_HELD);
		expect(gateCalls).toEqual([]);
		expect(await Bun.file(path.join(after.retainedDir ?? "", "work.txt")).text()).toBe("PreGate\n");
		const persisted = await Bun.file(path.join(path.dirname(after.retainedDir ?? ""), WORKSPACE_LEASE_FILE)).json();
		expect(persisted).toMatchObject(SHUTDOWN_HELD);
		await fs.rm(path.dirname(after.retainedDir ?? ""), { recursive: true, force: true });
	});

	it("bounds and holds an explicit release already in flight when shutdown starts", async () => {
		const { promise: gateDone, resolve: finishGate } = Promise.withResolvers<void>();
		gate(async () => {
			await gateDone;
			return { ok: true };
		});
		await spawn("ExplicitStuck");
		const before = lease("ExplicitStuck");
		const explicit = host.release("ExplicitStuck", { generation: before.generation, reason: "retire" });
		await AgentLifecycleManager.global().dispose(Date.now(), Date.now() + 300);
		expect(lease("ExplicitStuck")).toMatchObject(SHUTDOWN_HELD);
		expect(await Bun.file(path.join(before.baseDir, WORKSPACE_LEASE_FILE)).json()).toMatchObject(SHUTDOWN_HELD);
		finishGate();
		const result = await explicit;
		expect(result).toMatchObject({ status: "held", reason: SHUTDOWN_HELD.holdReason });
		expect(await Bun.file(path.join(result.lease?.retainedDir ?? "", "work.txt")).text()).toBe("ExplicitStuck\n");
		await fs.rm(path.dirname(result.lease?.retainedDir ?? ""), { recursive: true, force: true });
	});

	it("keeps the short grace for an adopted agent without a lease while gates are registered", async () => {
		gate(async () => ({ ok: true }));
		const { promise: never } = Promise.withResolvers<void>();
		const session = { dispose: () => never } as unknown as AgentSession;
		AgentRegistry.global().register({ id: "NoLease", displayName: "NoLease", kind: "sub", session, status: "idle" });
		AgentLifecycleManager.global().adopt("NoLease", { idleTtlMs: 0 });
		// The workspace bound stays at its 60 s default: waiting on it would exceed this test's 5 s timeout.
		await AgentLifecycleManager.global().dispose(Date.now() + 100);
		expect(host.list().some(l => l.agentId === "NoLease")).toBe(false);
	});

	it("does not extend a signal exit when no release gate is registered", async () => {
		await spawn("NoGateLease");
		const before = lease("NoGateLease");
		const register = vi.spyOn(postmortem, "register");
		await AgentLifecycleManager.global().dispose();
		expect(register.mock.calls.some(([id]) => id === "workspace-release-gates")).toBe(false);
		expect(lease("NoGateLease").state).toBe("released");
		expect(await exists(before.baseDir)).toBe(false);
	});

	it("holds a signal exit open for a gated unsettled release, bounded from the given deadline", async () => {
		const { promise: gateDone, resolve: finishGate } = Promise.withResolvers<void>();
		gate(async () => {
			await gateDone;
			return { ok: true };
		});
		await spawn("SignalHold");
		const register = vi.spyOn(postmortem, "register");
		await AgentLifecycleManager.global().dispose(Date.now(), Date.now() + 300);
		const hold = register.mock.calls.find(([id]) => id === "workspace-release-gates");
		// Remaining workspace bound (≤ 300 ms) plus the held-write headroom, announced to the user.
		expect(hold?.[2]?.waitMessage).toBe("Waiting for workspace release gates");
		expect(hold?.[2]?.deadlineMs).toBeGreaterThan(5000);
		expect(hold?.[2]?.deadlineMs).toBeLessThanOrEqual(5300);
		expect(lease("SignalHold")).toMatchObject(SHUTDOWN_HELD);
		const retained = retainedLease("SignalHold");
		finishGate();
		const after = await retained;
		await fs.rm(path.dirname(after.retainedDir ?? ""), { recursive: true, force: true });
	});
});

/** Pause the actual first sidecar rename, leaving subsequent writes to the real filesystem. */
function pauseLeaseRename(baseDir: string, synchronous = false): { entered: Promise<void>; resume: () => void } {
	const entered = Promise.withResolvers<void>();
	const resume = Promise.withResolvers<void>();
	const rename = fs.rename;
	if (synchronous) {
		// Real disk writes with deterministic completion: independent persistence is a barrier, not a timer.
		const write = Bun.write.bind(Bun);
		vi.spyOn(Bun, "write").mockImplementation((file, data, options) => {
			if (
				typeof file === "string" &&
				file.startsWith(root) &&
				file.includes(WORKSPACE_LEASE_FILE) &&
				typeof data === "string"
			) {
				fsSync.writeFileSync(file, data);
				return Promise.resolve(Buffer.byteLength(data));
			}
			return write(file, data, options);
		});
	}
	let paused = false;
	vi.spyOn(fs, "rename").mockImplementation(async (from, to) => {
		if (!paused && String(to) === path.join(baseDir, WORKSPACE_LEASE_FILE)) {
			paused = true;
			entered.resolve();
			await resume.promise;
		}
		if (synchronous && String(to).startsWith(root) && String(to).endsWith(WORKSPACE_LEASE_FILE)) {
			fsSync.renameSync(from, to);
		} else {
			await rename(from, to);
		}
	});
	return { entered: entered.promise, resume: resume.resolve };
}

function leaseInit(agentId: string, baseDir: string): Omit<WorkspaceLease, "generation" | "ownerPid" | "state"> {
	return {
		agentId,
		baseDir,
		workspaceDir: path.join(baseDir, "m"),
		repoRoot: repo,
		backend: "fixture",
		keepAlive: true,
		baseline: { commit: firstCommit, source: "working-tree", nested: "source-working-copy" },
	};
}

describe("workspace lease persistence ownership", () => {
	for (const action of ["supersede", "shutdown"] as const) {
		it(`denies teardown when ${action} occurs during the persistence drain`, async () => {
			const service = new WorkspaceLifecycleService();
			const baseDir = path.join(root, `drain-${action}`);
			await fs.mkdir(baseDir);
			const pause = pauseLeaseRename(baseDir);
			const generation = service.register(leaseInit(`Drain-${action}`, baseDir), async () => {});
			const permission = service.runGates(`Drain-${action}`, generation);
			await pause.entered;
			let held: Promise<void> | undefined;
			try {
				if (action === "supersede") {
					const nextBase = `${baseDir}-next`;
					await fs.mkdir(nextBase);
					service.register(leaseInit(`Drain-${action}`, nextBase), async () => {});
				}
				held = service.holdAtShutdown(`Drain-${action}`, "shutdown while persisting");
				pause.resume();
				expect(await permission).toBe(action === "supersede" ? "lease superseded" : "shutdown while persisting");
				await held;
			} finally {
				pause.resume();
				await permission;
				await held;
			}
		});
	}

	for (const reuseAgentId of [true, false]) {
		it(`orders reused-slot writes without blocking distinct paths (reuse agent id: ${reuseAgentId})`, async () => {
			const service = new WorkspaceLifecycleService();
			const baseDir = path.join(root, `ordered-${reuseAgentId}`);
			const independentBase = `${baseDir}-independent`;
			const barrierBase = `${baseDir}-barrier`;
			await fs.mkdir(baseDir);
			await fs.mkdir(independentBase);
			await fs.mkdir(barrierBase);
			const pause = pauseLeaseRename(baseDir, true);
			const oldId = `Ordered-${reuseAgentId}`;
			const old = service.register(leaseInit(oldId, baseDir), async () => {});
			service.update(oldId, old, { state: "held", holdReason: "old source retained" });
			const oldSettled = service.whenSettled(oldId, old);
			await pause.entered;
			let replacementHeld: Promise<void> | undefined;
			try {
				// Reusing an id on a distinct path must neither block that path nor cancel the old held write.
				const independent = service.register(leaseInit(oldId, independentBase), async () => {});
				await service.holdAtShutdown(oldId, "independent retained");
				expect(await Bun.file(path.join(independentBase, WORKSPACE_LEASE_FILE)).json()).toMatchObject({
					generation: independent,
					state: "held",
				});
				const nextId = reuseAgentId ? oldId : `${oldId}-other`;
				const next = service.register(leaseInit(nextId, baseDir), async () => {});
				replacementHeld = service.holdAtShutdown(nextId, "replacement retained");
				let replacementSettled = false;
				void replacementHeld.then(() => {
					replacementSettled = true;
				});
				// This independent lease drains the same synchronous I/O steps after the successor.
				// A settled successor must not still be vulnerable to the paused predecessor's rename.
				const barrierId = `${oldId}-barrier`;
				service.register(leaseInit(barrierId, barrierBase), async () => {});
				await service.holdAtShutdown(barrierId, "barrier retained");
				expect(replacementSettled).toBe(false);
				pause.resume();
				await Promise.all([oldSettled, replacementHeld]);
				expect(await Bun.file(path.join(baseDir, WORKSPACE_LEASE_FILE)).json()).toMatchObject({
					generation: next,
					agentId: nextId,
					state: "held",
					holdReason: "replacement retained",
				});
				expect(await Bun.file(path.join(independentBase, WORKSPACE_LEASE_FILE)).json()).toMatchObject({
					generation: independent,
					state: "held",
				});
			} finally {
				pause.resume();
				await oldSettled;
				await replacementHeld;
			}
		});
	}
});

describe("task schema baseline", () => {
	const call = { agent: "task", task: "t", solutionSpace: "s", isolated: true, baseline: "abc123" };
	for (const options of [
		{ isolationEnabled: true, batchEnabled: false },
		{ isolationEnabled: true, batchEnabled: true },
		{ isolationEnabled: true, batchEnabled: false, effortEnabled: true },
		{ isolationEnabled: true, batchEnabled: true, effortEnabled: true },
	]) {
		it(`keeps baseline (${JSON.stringify(options)})`, () => {
			const input = options.batchEnabled ? { context: "c", tasks: [call] } : call;
			const parsed = getTaskSchema(options)(input);
			expect(parsed instanceof type.errors).toBe(false);
			expect(parsed).toMatchObject(
				options.batchEnabled ? { tasks: [{ baseline: "abc123" }] } : { baseline: "abc123" },
			);
		});
	}

	it("strips baseline when isolation is disabled", () => {
		const parsed = getTaskSchema({ isolationEnabled: false, batchEnabled: false })(call);
		expect(parsed).not.toHaveProperty("baseline");
	});
});

// Keep the lifecycle manager from leaking adopted agents into other files.
afterAll(async () => {
	await AgentLifecycleManager.global().dispose();
	expect(
		getWorkspaceLifecycleService()
			.list()
			.every(l => l.state === "released" || l.state === "held"),
	).toBe(true);
});
