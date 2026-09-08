import { describe, expect, it } from "bun:test";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { pathToFileURL } from "node:url";
import { WorkerCore } from "@oh-my-pi/pi-coding-agent/eval/js/worker-core";
import type {
	SessionSnapshot,
	Transport,
	WorkerInbound,
	WorkerOutbound,
} from "@oh-my-pi/pi-coding-agent/eval/js/worker-protocol";
import { postmortem } from "@oh-my-pi/pi-utils";

interface WorkerHarness {
	send(message: WorkerInbound): void;
	onMessage(handler: (message: WorkerOutbound) => void): () => void;
}

function createWorkerHarness(): WorkerHarness {
	const hostListeners = new Set<(message: WorkerOutbound) => void>();
	const workerListeners = new Set<(message: WorkerInbound) => void>();
	const transport: Transport = {
		send: message => {
			queueMicrotask(() => {
				for (const listener of hostListeners) listener(message);
			});
		},
		onMessage: handler => {
			workerListeners.add(handler);
			return () => workerListeners.delete(handler);
		},
		close: () => {},
	};
	new WorkerCore(transport, {
		mode: "inline",
		interceptUnhandledRejections: postmortem.interceptUnhandledRejections,
	});
	return {
		send(message) {
			queueMicrotask(() => {
				for (const listener of workerListeners) listener(message);
			});
		},
		onMessage(handler) {
			hostListeners.add(handler);
			return () => hostListeners.delete(handler);
		},
	};
}

function waitForMessage(
	harness: WorkerHarness,
	predicate: (message: WorkerOutbound) => boolean,
): Promise<WorkerOutbound> {
	const { promise, resolve } = Promise.withResolvers<WorkerOutbound>();
	let unsubscribe = (): void => {};
	unsubscribe = harness.onMessage(message => {
		if (!predicate(message)) return;
		unsubscribe();
		resolve(message);
	});
	return promise;
}

async function initializeWorker(harness: WorkerHarness, snapshot: SessionSnapshot): Promise<void> {
	const ready = waitForMessage(harness, message => message.type === "ready");
	harness.send({ type: "init", snapshot });
	expect((await ready).type).toBe("ready");
}

function installFatalCapture(): {
	fatal: unknown[];
	uninstall: () => void;
} {
	const fatal: unknown[] = [];
	const onUnhandled = (reason: unknown): void => {
		fatal.push(reason);
	};
	const onUncaught = (err: Error): void => {
		fatal.push(err);
	};
	process.on("unhandledRejection", onUnhandled);
	process.on("uncaughtException", onUncaught);
	return {
		fatal,
		uninstall: () => {
			process.off("unhandledRejection", onUnhandled);
			process.off("uncaughtException", onUncaught);
		},
	};
}

describe("WorkerCore", () => {
	it("retains a finished cell until its floated bridge promise settles across cells", async () => {
		const harness = createWorkerHarness();
		const cwd = process.cwd();
		const snapshot = { cwd, sessionId: "floated-bridge", localRoots: {} };
		await initializeWorker(harness, snapshot);

		const outbound: WorkerOutbound[] = [];
		const unsubscribe = harness.onMessage(message => outbound.push(message));
		try {
			const toolCall = waitForMessage(harness, message => message.type === "tool-call");
			const firstResult = waitForMessage(
				harness,
				message => message.type === "result" && message.runId === "floated-first",
			);
			harness.send({
				type: "run",
				runId: "floated-first",
				filename: "[floated-first].js",
				snapshot,
				code: "globalThis.__worker_core_saved = globalThis.__omp_call_tool__('fake', {}); 'returned';",
			});

			const call = (await toolCall) as Extract<WorkerOutbound, { type: "tool-call" }>;
			await Bun.sleep(10);
			expect(outbound).not.toContainEqual(expect.objectContaining({ type: "result", runId: "floated-first" }));

			harness.send({ type: "tool-reply", id: call.id, reply: { ok: true, value: "bridge-result" } });
			expect(await firstResult).toMatchObject({ type: "result", runId: "floated-first", ok: true });

			const secondText = waitForMessage(
				harness,
				message => message.type === "text" && message.runId === "floated-second",
			);
			const secondResult = waitForMessage(
				harness,
				message => message.type === "result" && message.runId === "floated-second",
			);
			harness.send({
				type: "run",
				runId: "floated-second",
				filename: "[floated-second].js",
				snapshot,
				code: "display(await globalThis.__worker_core_saved);",
			});
			expect(await secondText).toMatchObject({ type: "text", runId: "floated-second", chunk: "bridge-result\n" });
			expect(await secondResult).toMatchObject({ type: "result", runId: "floated-second", ok: true });
		} finally {
			unsubscribe();
			harness.send({ type: "close" });
		}
	});

	it("preserves a delayed bridge error for a later cell without hanging", async () => {
		const harness = createWorkerHarness();
		const snapshot = { cwd: process.cwd(), sessionId: "floated-error", localRoots: {} };
		await initializeWorker(harness, snapshot);
		try {
			const toolCall = waitForMessage(harness, message => message.type === "tool-call");
			const firstResult = waitForMessage(
				harness,
				message => message.type === "result" && message.runId === "error-first",
			);
			harness.send({
				type: "run",
				runId: "error-first",
				filename: "[error-first].js",
				snapshot,
				code: "globalThis.__worker_core_error = globalThis.__omp_call_tool__('fake', {}).catch(error => error.message); 'returned';",
			});
			const call = (await toolCall) as Extract<WorkerOutbound, { type: "tool-call" }>;
			harness.send({
				type: "tool-reply",
				id: call.id,
				reply: { ok: false, error: { name: "BridgeError", message: "delayed failure" } },
			});
			expect(await firstResult).toMatchObject({ type: "result", runId: "error-first", ok: true });

			const secondText = waitForMessage(
				harness,
				message => message.type === "text" && message.runId === "error-second",
			);
			const secondResult = waitForMessage(
				harness,
				message => message.type === "result" && message.runId === "error-second",
			);
			harness.send({
				type: "run",
				runId: "error-second",
				filename: "[error-second].js",
				snapshot,
				code: "display(await globalThis.__worker_core_error);",
			});
			expect(await secondText).toMatchObject({ type: "text", runId: "error-second", chunk: "delayed failure\n" });
			expect(await secondResult).toMatchObject({ type: "result", runId: "error-second", ok: true });
		} finally {
			harness.send({ type: "close" });
		}
	});

	it("drains bridge calls started by floated promise continuations before settling once", async () => {
		const harness = createWorkerHarness();
		const snapshot = { cwd: process.cwd(), sessionId: "chained-bridge", localRoots: {} };
		await initializeWorker(harness, snapshot);
		const outbound: WorkerOutbound[] = [];
		const unsubscribe = harness.onMessage(message => outbound.push(message));
		try {
			const firstCall = waitForMessage(harness, message => message.type === "tool-call" && message.name === "first");
			const result = waitForMessage(harness, message => message.type === "result" && message.runId === "chain");
			harness.send({
				type: "run",
				runId: "chain",
				filename: "[chain].js",
				snapshot,
				code: "globalThis.__worker_core_chain = globalThis.__omp_call_tool__('first', {}).then(() => globalThis.__omp_call_tool__('second', {})); 'returned';",
			});
			const first = (await firstCall) as Extract<WorkerOutbound, { type: "tool-call" }>;
			const secondCall = waitForMessage(
				harness,
				message => message.type === "tool-call" && message.name === "second",
			);
			harness.send({ type: "tool-reply", id: first.id, reply: { ok: true, value: "first-result" } });
			const second = (await secondCall) as Extract<WorkerOutbound, { type: "tool-call" }>;
			expect(outbound).not.toContainEqual(expect.objectContaining({ type: "result", runId: "chain" }));
			harness.send({ type: "tool-reply", id: second.id, reply: { ok: true, value: "second-result" } });
			expect(await result).toMatchObject({ type: "result", runId: "chain", ok: true });
			const lateReplyLog = waitForMessage(
				harness,
				message => message.type === "log" && message.msg === "Ignored unmatched JS eval tool reply",
			);
			harness.send({ type: "tool-reply", id: second.id, reply: { ok: true, value: "duplicate" } });
			expect(await lateReplyLog).toMatchObject({ type: "log", level: "warn", meta: { id: second.id } });
			await Bun.sleep(10);
			expect(outbound.filter(message => message.type === "result" && message.runId === "chain")).toHaveLength(1);
			expect(outbound.filter(message => message.type === "tool-call" && message.runId === "chain")).toHaveLength(2);
		} finally {
			unsubscribe();
			harness.send({ type: "close" });
		}
	});

	it("closes a run with a pending floated bridge call without a late result", async () => {
		const harness = createWorkerHarness();
		const snapshot = { cwd: process.cwd(), sessionId: "close-bridge", localRoots: {} };
		await initializeWorker(harness, snapshot);
		const outbound: WorkerOutbound[] = [];
		harness.onMessage(message => outbound.push(message));
		const toolCall = waitForMessage(harness, message => message.type === "tool-call");
		harness.send({
			type: "run",
			runId: "close-pending",
			filename: "[close-pending].js",
			snapshot,
			code: "globalThis.__omp_call_tool__('fake', {}).catch(() => undefined); 'returned';",
		});
		const call = (await toolCall) as Extract<WorkerOutbound, { type: "tool-call" }>;
		const closed = waitForMessage(harness, message => message.type === "closed");
		harness.send({ type: "close" });
		expect((await closed).type).toBe("closed");
		harness.send({ type: "tool-reply", id: call.id, reply: { ok: true, value: "late" } });
		await Bun.sleep(10);
		expect(outbound.filter(message => message.type === "result" && message.runId === "close-pending")).toEqual([]);
		expect(outbound.filter(message => message.type === "closed")).toHaveLength(1);
	});

	it("reports same-realm cwd conflicts through the worker protocol", async () => {
		const first = createWorkerHarness();
		const second = createWorkerHarness();
		const cwd = process.cwd();
		await initializeWorker(first, { cwd, sessionId: "same-realm-first", localRoots: {} });
		await initializeWorker(second, { cwd, sessionId: "same-realm-second", localRoots: {} });

		const gate = Promise.withResolvers<void>();
		const entered = Promise.withResolvers<void>();
		(globalThis as { __omp_worker_core_gate?: { entered(): void; wait: Promise<void> } }).__omp_worker_core_gate = {
			entered: () => entered.resolve(),
			wait: gate.promise,
		};
		try {
			first.send({
				type: "run",
				runId: "hold-first-runtime",
				code: "globalThis.__omp_worker_core_gate.entered(); await globalThis.__omp_worker_core_gate.wait;",
				filename: "[same-realm-first].js",
				snapshot: { cwd, sessionId: "same-realm-first", localRoots: {} },
			});
			await entered.promise;

			const result = waitForMessage(
				second,
				message => message.type === "result" && message.runId === "overlap-second-runtime",
			);
			second.send({
				type: "run",
				runId: "overlap-second-runtime",
				code: "1 + 1;",
				filename: "[same-realm-second].js",
				snapshot: { cwd, sessionId: "same-realm-second", localRoots: {} },
			});

			expect(await result).toMatchObject({
				type: "result",
				runId: "overlap-second-runtime",
				ok: false,
				error: { message: "Cannot run code while another same-realm JS runtime is running" },
			});
		} finally {
			gate.resolve();
			delete (globalThis as { __omp_worker_core_gate?: { entered(): void; wait: Promise<void> } })
				.__omp_worker_core_gate;
			first.send({ type: "close" });
			second.send({ type: "close" });
		}
	});

	it("re-init while a same-realm run is live does not crash the process", async () => {
		const first = createWorkerHarness();
		const second = createWorkerHarness();
		const cwd = process.cwd();
		await initializeWorker(first, { cwd, sessionId: "reinit-first", localRoots: {} });
		await initializeWorker(second, { cwd, sessionId: "reinit-second", localRoots: {} });

		const gate = Promise.withResolvers<void>();
		const entered = Promise.withResolvers<void>();
		(globalThis as { __omp_worker_core_gate?: { entered(): void; wait: Promise<void> } }).__omp_worker_core_gate = {
			entered: () => entered.resolve(),
			wait: gate.promise,
		};

		const { fatal, uninstall } = installFatalCapture();
		try {
			first.send({
				type: "run",
				runId: "hold-for-reinit",
				code: "globalThis.__omp_worker_core_gate.entered(); await globalThis.__omp_worker_core_gate.wait;",
				filename: "[reinit-first].js",
				snapshot: { cwd, sessionId: "reinit-first", localRoots: {} },
			});
			await entered.promise;

			// Re-init the second core while the first still owns the realm. Production
			// inline workers deliver this on a microtask; a setCwd throw here used to
			// become a process-fatal unhandledRejection / uncaughtException.
			const reinit = waitForMessage(second, message => message.type === "ready" || message.type === "init-failed");
			second.send({ type: "init", snapshot: { cwd, sessionId: "reinit-second", localRoots: {} } });
			const reply = await reinit;
			expect(reply.type).toBe("ready");

			// Concurrent run still fails at the exclusive run boundary, via protocol.
			const result = waitForMessage(
				second,
				message => message.type === "result" && message.runId === "overlap-after-reinit",
			);
			second.send({
				type: "run",
				runId: "overlap-after-reinit",
				code: "1 + 1;",
				filename: "[reinit-second].js",
				snapshot: { cwd, sessionId: "reinit-second", localRoots: {} },
			});
			expect(await result).toMatchObject({
				type: "result",
				runId: "overlap-after-reinit",
				ok: false,
				error: { message: "Cannot run code while another same-realm JS runtime is running" },
			});

			// Drain microtasks so a latent fatal would surface.
			await Bun.sleep(0);
			expect(fatal).toEqual([]);
		} finally {
			uninstall();
			gate.resolve();
			delete (globalThis as { __omp_worker_core_gate?: { entered(): void; wait: Promise<void> } })
				.__omp_worker_core_gate;
			first.send({ type: "close" });
			second.send({ type: "close" });
		}
	});

	it("concurrent inits under a live same-realm run stay process-safe", async () => {
		const first = createWorkerHarness();
		const second = createWorkerHarness();
		const third = createWorkerHarness();
		const cwd = process.cwd();
		await initializeWorker(first, { cwd, sessionId: "init-live-first", localRoots: {} });
		await initializeWorker(second, { cwd, sessionId: "init-live-second", localRoots: {} });
		await initializeWorker(third, { cwd, sessionId: "init-live-third", localRoots: {} });

		const gate = Promise.withResolvers<void>();
		const entered = Promise.withResolvers<void>();
		(globalThis as { __omp_worker_core_gate?: { entered(): void; wait: Promise<void> } }).__omp_worker_core_gate = {
			entered: () => entered.resolve(),
			wait: gate.promise,
		};

		const { fatal, uninstall } = installFatalCapture();
		try {
			first.send({
				type: "run",
				runId: "hold-for-multi-init",
				code: "globalThis.__omp_worker_core_gate.entered(); await globalThis.__omp_worker_core_gate.wait;",
				filename: "[init-live-first].js",
				snapshot: { cwd, sessionId: "init-live-first", localRoots: {} },
			});
			await entered.promise;

			const readySecond = waitForMessage(
				second,
				message => message.type === "ready" || message.type === "init-failed",
			);
			const readyThird = waitForMessage(
				third,
				message => message.type === "ready" || message.type === "init-failed",
			);
			second.send({ type: "init", snapshot: { cwd, sessionId: "init-live-second", localRoots: {} } });
			third.send({ type: "init", snapshot: { cwd, sessionId: "init-live-third", localRoots: {} } });
			expect((await readySecond).type).toBe("ready");
			expect((await readyThird).type).toBe("ready");

			const resultSecond = waitForMessage(
				second,
				message => message.type === "result" && message.runId === "overlap-second",
			);
			const resultThird = waitForMessage(
				third,
				message => message.type === "result" && message.runId === "overlap-third",
			);
			second.send({
				type: "run",
				runId: "overlap-second",
				code: "2",
				filename: "[init-live-second].js",
				snapshot: { cwd, sessionId: "init-live-second", localRoots: {} },
			});
			third.send({
				type: "run",
				runId: "overlap-third",
				code: "3",
				filename: "[init-live-third].js",
				snapshot: { cwd, sessionId: "init-live-third", localRoots: {} },
			});
			expect(await resultSecond).toMatchObject({
				type: "result",
				runId: "overlap-second",
				ok: false,
				error: { message: "Cannot run code while another same-realm JS runtime is running" },
			});
			expect(await resultThird).toMatchObject({
				type: "result",
				runId: "overlap-third",
				ok: false,
				error: { message: "Cannot run code while another same-realm JS runtime is running" },
			});

			await Bun.sleep(0);
			expect(fatal).toEqual([]);
		} finally {
			uninstall();
			gate.resolve();
			delete (globalThis as { __omp_worker_core_gate?: { entered(): void; wait: Promise<void> } })
				.__omp_worker_core_gate;
			first.send({ type: "close" });
			second.send({ type: "close" });
			third.send({ type: "close" });
		}
	});

	it("first init while a same-realm run is live fails via init-failed and recovers", async () => {
		const first = createWorkerHarness();
		const second = createWorkerHarness(); // never initialized: no runtime exists yet
		const cwd = process.cwd();
		await initializeWorker(first, { cwd, sessionId: "first-init-live-first", localRoots: {} });

		const gate = Promise.withResolvers<void>();
		const entered = Promise.withResolvers<void>();
		(globalThis as { __omp_worker_core_gate?: { entered(): void; wait: Promise<void> } }).__omp_worker_core_gate = {
			entered: () => entered.resolve(),
			wait: gate.promise,
		};

		const { fatal, uninstall } = installFatalCapture();
		try {
			const firstText = waitForMessage(
				first,
				message => message.type === "text" && message.runId === "hold-for-first-init",
			);
			const firstResult = waitForMessage(
				first,
				message => message.type === "result" && message.runId === "hold-for-first-init",
			);
			first.send({
				type: "run",
				runId: "hold-for-first-init",
				code: "globalThis.__omp_worker_core_gate.entered(); await globalThis.__omp_worker_core_gate.wait; __omp_session__.sessionId;",
				filename: "[first-init-live-first].js",
				snapshot: { cwd, sessionId: "first-init-live-first", localRoots: {} },
			});
			await entered.promise;

			// A fresh runtime's install would Object.assign over the live runtime's
			// globals mid-run; it must fail via the protocol instead.
			const reply = waitForMessage(second, message => message.type === "ready" || message.type === "init-failed");
			second.send({ type: "init", snapshot: { cwd, sessionId: "first-init-live-second", localRoots: {} } });
			expect(await reply).toMatchObject({
				type: "init-failed",
				error: { message: "Cannot initialize a JS runtime while another same-realm JS runtime is running" },
			});

			// The held run's globals were not clobbered: it still resolves its own
			// session bag and completes cleanly.
			gate.resolve();
			expect(await firstText).toMatchObject({
				type: "text",
				runId: "hold-for-first-init",
				chunk: "first-init-live-first\n",
			});
			expect(await firstResult).toMatchObject({ type: "result", runId: "hold-for-first-init", ok: true });

			// Once the realm is free, the same core initializes cleanly.
			await initializeWorker(second, { cwd, sessionId: "first-init-live-second", localRoots: {} });

			// Drain the microtask queue so any latent fatal would surface.
			for (let i = 0; i < 8; i++) await Promise.resolve();
			expect(fatal).toEqual([]);
		} finally {
			uninstall();
			gate.resolve();
			delete (globalThis as { __omp_worker_core_gate?: { entered(): void; wait: Promise<void> } })
				.__omp_worker_core_gate;
			first.send({ type: "close" });
			second.send({ type: "close" });
		}
	});

	it("keeps the process cwd while another cell is mid-run", async () => {
		const dirA = await fs.mkdtemp(path.join(os.tmpdir(), "omp-cwd-a-"));
		const dirB = await fs.mkdtemp(path.join(os.tmpdir(), "omp-cwd-b-"));
		const chdirs: string[] = [];
		const hostListeners = new Set<(message: WorkerOutbound) => void>();
		const workerListeners = new Set<(message: WorkerInbound) => void>();
		const transport: Transport = {
			send: message => {
				queueMicrotask(() => {
					for (const listener of hostListeners) listener(message);
				});
			},
			onMessage: handler => {
				workerListeners.add(handler);
				return () => workerListeners.delete(handler);
			},
			close: () => {},
		};
		new WorkerCore(transport, { mode: "isolated", chdir: cwd => chdirs.push(cwd) });
		const harness: WorkerHarness = {
			send(message) {
				queueMicrotask(() => {
					for (const listener of workerListeners) listener(message);
				});
			},
			onMessage(handler) {
				hostListeners.add(handler);
				return () => hostListeners.delete(handler);
			},
		};

		const gate = Promise.withResolvers<void>();
		const entered = Promise.withResolvers<void>();
		(globalThis as { __omp_worker_cwd_gate?: { entered(): void; wait: Promise<void> } }).__omp_worker_cwd_gate = {
			entered: () => entered.resolve(),
			wait: gate.promise,
		};
		try {
			await initializeWorker(harness, { cwd: dirA, sessionId: "cwd-race", localRoots: {} });
			expect(chdirs).toEqual([dirA]);

			const holdResult = waitForMessage(
				harness,
				message => message.type === "result" && message.runId === "cwd-hold",
			);
			harness.send({
				type: "run",
				runId: "cwd-hold",
				code: "globalThis.__omp_worker_cwd_gate.entered(); await globalThis.__omp_worker_cwd_gate.wait;",
				filename: "[cwd-race-hold].js",
				snapshot: { cwd: dirA, sessionId: "cwd-race", localRoots: {} },
			});
			await entered.promise;

			// A second cell with a different cwd while the first is suspended must
			// not move the realm-wide process cwd out from under the live cell.
			const skipLog = waitForMessage(
				harness,
				message => message.type === "log" && message.msg.includes("kept its process cwd"),
			);
			const overlapResult = waitForMessage(
				harness,
				message => message.type === "result" && message.runId === "cwd-overlap",
			);
			harness.send({
				type: "run",
				runId: "cwd-overlap",
				code: "1 + 1;",
				filename: "[cwd-race-overlap].js",
				snapshot: { cwd: dirB, sessionId: "cwd-race", localRoots: {} },
			});
			expect(await overlapResult).toMatchObject({ type: "result", runId: "cwd-overlap", ok: true });
			expect(chdirs).not.toContain(dirB);
			await skipLog;

			gate.resolve();
			expect(await holdResult).toMatchObject({ type: "result", runId: "cwd-hold", ok: true });

			// With the realm quiet again, the next cell lands the deferred move.
			const soloResult = waitForMessage(
				harness,
				message => message.type === "result" && message.runId === "cwd-solo",
			);
			harness.send({
				type: "run",
				runId: "cwd-solo",
				code: "2 + 2;",
				filename: "[cwd-race-solo].js",
				snapshot: { cwd: dirB, sessionId: "cwd-race", localRoots: {} },
			});
			expect(await soloResult).toMatchObject({ type: "result", runId: "cwd-solo", ok: true });
			expect(chdirs.at(-1)).toBe(dirB);
		} finally {
			gate.resolve();
			delete (globalThis as { __omp_worker_cwd_gate?: { entered(): void; wait: Promise<void> } })
				.__omp_worker_cwd_gate;
			harness.send({ type: "close" });
			await fs.rm(dirA, { recursive: true, force: true });
			await fs.rm(dirB, { recursive: true, force: true });
		}
	});

	it("folds a floated delayed bridge rejection before the isolated worker result", async () => {
		const workerCoreUrl = pathToFileURL(path.resolve(import.meta.dir, "../../src/eval/js/worker-core.ts")).href;
		const probe = `import { WorkerCore } from ${JSON.stringify(workerCoreUrl)};

const inbound = new Set();
let core;
let watchdog;
const transport = {
	send(message) {
		if (message.type === "tool-call") {
			queueMicrotask(() => {
				for (const listener of inbound) listener({
					type: "tool-reply",
					id: message.id,
					reply: { ok: false, error: { name: "BridgeError", message: "floated failure" } },
				});
			});
		}
		if (message.type === "result" && message.runId === "floated-rejection") {
			process.stdout.write(JSON.stringify(message));
			process.exitCode = message.ok ? 2 : 0;
			clearTimeout(watchdog);
			core.dispose();
		}
	},
	onMessage(handler) { inbound.add(handler); return () => inbound.delete(handler); },
	close() {},
};
core = new WorkerCore(transport, { mode: "isolated" });
const snapshot = { cwd: process.cwd(), sessionId: "floated-rejection", localRoots: {}, preludes: [] };
watchdog = setTimeout(() => process.exit(3), 2000);
for (const listener of inbound) listener({ type: "init", snapshot });
for (const listener of inbound) listener({
	type: "run",
	runId: "floated-rejection",
	filename: "[floated-rejection].js",
	snapshot,
	code: "globalThis.__omp_call_tool__('fake', {}); 'returned';",
});
`;
		const root = await fs.mkdtemp(path.join(os.tmpdir(), "omp-floated-rejection-"));
		const probePath = path.join(root, "probe.ts");
		try {
			await Bun.write(probePath, probe);
			const proc = Bun.spawn([process.execPath, probePath], {
				cwd: process.cwd(),
				stdout: "pipe",
				stderr: "pipe",
				env: { ...process.env },
			});
			const [stdout, stderr, exitCode] = await Promise.all([
				new Response(proc.stdout).text(),
				new Response(proc.stderr).text(),
				proc.exited,
			]);
			expect(exitCode).toBe(0);
			expect(JSON.parse(stdout.trim())).toMatchObject({
				type: "result",
				runId: "floated-rejection",
				ok: false,
				error: { name: "BridgeError", message: "Unhandled rejection (missing await?): floated failure" },
			});
			expect(stderr).toBe("");
		} finally {
			await fs.rm(root, { recursive: true, force: true });
		}
	});

	it("survives concurrent same-realm setCwd in a child process with postmortem loaded", async () => {
		// Process-level oracle: the production crash was postmortem killing the process
		// after an unhandled rejection from concurrent inline setCwd. This must stay green
		// even when postmortem's fatal handlers are installed.
		const postmortemUrl = pathToFileURL(path.resolve(import.meta.dir, "../../../utils/src/postmortem.ts")).href;
		const runtimeUrl = pathToFileURL(path.resolve(import.meta.dir, "../../src/eval/js/shared/runtime.ts")).href;

		const probe = `import { pathToFileURL } from "node:url";

await import(${JSON.stringify(postmortemUrl)});
const { JsRuntime } = await import(${JSON.stringify(runtimeUrl)});

const first = new JsRuntime({ initialCwd: process.cwd(), sessionId: "child-first" });
const second = new JsRuntime({ initialCwd: process.cwd(), sessionId: "child-second" });
const gate = Promise.withResolvers();
const entered = Promise.withResolvers();

const hooks = {
	onText() {},
	onDisplay() {},
	callTool: async () => undefined,
};

second.setRunScope({ gate: gate.promise, entered: () => entered.resolve() });
const hold = second.run("entered(); await gate;", "[child-second].js", hooks);
await entered.promise;

// Historical crash path: concurrent setCwd while another same-realm runtime is live.
first.setCwd(process.cwd() + "/child-pending");
second.setCwd(process.cwd());

// Microtask delivery must not become process-fatal either.
queueMicrotask(() => {
	first.setCwd(process.cwd() + "/child-pending-2");
});
await Promise.resolve();
await Bun.sleep(0);

gate.resolve();
await hold;
first.dispose();
second.dispose();
console.log("survived concurrent setCwd");
process.exit(0);
`;

		const root = await fs.mkdtemp(path.join(os.tmpdir(), "omp-same-realm-"));
		const probePath = path.join(root, "probe.ts");
		try {
			await Bun.write(probePath, probe);
			const proc = Bun.spawn([process.execPath, probePath], {
				cwd: process.cwd(),
				stdout: "pipe",
				stderr: "pipe",
				env: { ...process.env },
			});
			// Real process liveness cannot use fake timers. Bound a wedged child, but
			// clear the watchdog on the normal path so it never becomes a fixed wait.
			const watchdog = setTimeout(() => {
				try {
					proc.kill("SIGKILL");
				} catch {}
			}, 5000);
			try {
				const [stdout, stderr, exitCode] = await Promise.all([
					new Response(proc.stdout).text(),
					new Response(proc.stderr).text(),
					proc.exited,
				]);
				expect(exitCode).toBe(0);
				expect(stdout).toContain("survived concurrent setCwd");
				expect(stderr).not.toContain("[Unhandled Rejection]");
				expect(stderr).not.toContain("[Uncaught Exception]");
				expect(stderr).not.toContain("another same-realm JS runtime is running");
			} finally {
				clearTimeout(watchdog);
			}
		} finally {
			await fs.rm(root, { recursive: true, force: true });
		}
	});
});
