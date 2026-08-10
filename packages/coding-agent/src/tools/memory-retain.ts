import { type } from "@oh-my-pi/omptype";
import type { AgentTool, AgentToolResult } from "@oh-my-pi/pi-agent-core";
import type { MemoryBackendSaveResult } from "../memory-backend/types";
import retainDescription from "../prompts/tools/retain.md" with { type: "text" };
import type { ToolSession } from ".";

const memoryRetainSchema = type({
	items: type({
		content: type("string").describe("information to remember"),
		"context?": type("string").describe("source context"),
	})
		.array()
		.atLeastLength(1)
		.describe("memories to retain"),
});

export type MemoryRetainParams = typeof memoryRetainSchema.infer;
export class MemoryRetainTool implements AgentTool<typeof memoryRetainSchema> {
	readonly name = "retain";
	readonly approval = "read" as const;
	readonly label = "Retain";
	readonly description = retainDescription;
	readonly parameters = memoryRetainSchema;
	readonly strict = true;
	readonly loadMode = "discoverable";
	readonly summary = "Store important facts in long-term memory";

	constructor(private readonly session: ToolSession) {}

	static createIf(session: ToolSession): MemoryRetainTool | null {
		const backend = session.getMemoryBackend?.()?.id ?? session.settings.get("memory.backend");
		if (backend !== "hindsight" && backend !== "mnemopi" && backend !== "supermemory") return null;
		return new MemoryRetainTool(session);
	}

	async execute(_id: string, params: MemoryRetainParams): Promise<AgentToolResult> {
		const backend = this.session.getMemoryBackend?.()?.id ?? this.session.settings.get("memory.backend");
		if (backend === "supermemory") {
			const memory = this.session.getMemoryRuntime?.();
			if (!memory) throw new Error("Supermemory backend is not initialised for this session.");
			const results: MemoryBackendSaveResult[] = [];
			for (const item of params.items) {
				results.push(
					await memory.save({ content: item.content, context: item.context, source: "coding-agent-retain" }),
				);
			}
			const stored = results.reduce((total, result) => total + result.stored, 0);
			const failures = results
				.map((result, index) =>
					result.stored > 0 ? undefined : `item ${index + 1}: ${result.message ?? "not stored"}`,
				)
				.filter((failure): failure is string => failure !== undefined);
			if (failures.length > 0) {
				throw new Error(
					stored === 0
						? failures.join("; ")
						: `Supermemory stored ${stored} of ${params.items.length} memories; failed ${failures.join("; ")}.`,
				);
			}
			const noun = stored === 1 ? "memory" : "memories";
			return {
				content: [{ type: "text", text: `${stored} ${noun} stored.` }],
				details: { count: stored },
			};
		}

		if (backend === "mnemopi") {
			const state = this.session.getMnemopiSessionState?.();
			if (!state) {
				throw new Error("Mnemopi backend is not initialised for this session.");
			}

			for (const item of params.items) {
				state.rememberScoped(item.content, {
					source: "coding-agent-retain",
					importance: 0.75,
					metadata: {
						session_id: state.sessionId,
						cwd: state.session.sessionManager.getCwd(),
						context: item.context ?? null,
						tool: "retain",
					},
					scope: "bank",
					extract: true,
					extractEntities: true,
					veracity: "tool",
					memoryType: "fact",
				});
			}

			const count = params.items.length;
			const noun = count === 1 ? "memory" : "memories";
			return {
				content: [{ type: "text", text: `${count} ${noun} stored.` }],
				details: { count },
			};
		}

		const state = this.session.getHindsightSessionState?.();
		if (!state) {
			throw new Error("Hindsight backend is not initialised for this session.");
		}

		// Push every item onto the session-owned queue and return immediately.
		// The queue flushes either when it reaches its batch threshold or when
		// its debounce timer fires. If the eventual batch fails, the queue
		// surfaces a UI-only warning notice — the LLM is not informed.
		for (const item of params.items) {
			state.enqueueRetain(item.content, item.context);
		}

		const count = params.items.length;
		const noun = count === 1 ? "memory" : "memories";
		return {
			content: [{ type: "text", text: `${count} ${noun} queued.` }],
			details: { count },
		};
	}
}
