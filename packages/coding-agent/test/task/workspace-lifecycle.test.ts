import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "bun:test";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { type } from "@oh-my-pi/omptype";
import { AgentLifecycleManager } from "@oh-my-pi/pi-coding-agent/registry/agent-lifecycle";
import { AgentRegistry } from "@oh-my-pi/pi-coding-agent/registry/agent-registry";
import type { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import * as executorModule from "@oh-my-pi/pi-coding-agent/task/executor";
import { prepareIsolationContext, runIsolatedSubprocess } from "@oh-my-pi/pi-coding-agent/task/isolation-runner";
import { getTaskSchema } from "@oh-my-pi/pi-coding-agent/task/types";
import {
	getWorkspaceLifecycleHost,
	getWorkspaceLifecycleService,
	WORKSPACE_LEASE_FILE,
	type WorkspaceLease,
	type WorkspaceReleaseGate,
} from "@oh-my-pi/pi-coding-agent/task/workspace-lifecycle";
import * as worktreeModule from "@oh-my-pi/pi-coding-agent/task/worktree";
import * as natives from "@oh-my-pi/pi-natives";
import { setWorktreesDir } from "@oh-my-pi/pi-utils";
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
}

/** Real isolation runner + lifecycle; only the model subprocess is faked. */
async function spawn(id: string, opts: SpawnOptions = {}): Promise<SingleResult> {
	const keepAlive = opts.keepAlive ?? true;
	const artifactsDir = path.join(root, "artifacts");
	await fs.mkdir(artifactsDir, { recursive: true });
	const session = {
		prepareForHeadlessAdvisorDrain: () => {},
		waitForAdvisorCatchup: async () => true,
		dispose: async () => {},
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
		preferredBackend: undefined,
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
