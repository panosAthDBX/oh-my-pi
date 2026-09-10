/** Session memory backend lifecycle and transcript resets. */

import type { Agent, AgentTool } from "@oh-my-pi/pi-agent-core";
import { logger } from "@oh-my-pi/pi-utils";
import type { ModelRegistry } from "../config/model-registry";
import type { Settings } from "../config/settings";
import type { HindsightSessionState } from "../hindsight/state";
import { resolveMemoryBackend } from "../memory-backend/resolve";
import type { MemoryBackend, MemoryBackendStartOptions } from "../memory-backend/types";
import type { MnemopiSessionState } from "../mnemopi/state";

/** Capabilities borrowed from the owning AgentSession. */
export interface PreparedMemoryBackendTransition {
	previousBackend: MemoryBackend;
}

export interface SessionMemoryHost {
	agent: Agent;
	settings: Settings;
	modelRegistry: ModelRegistry;
	isDisposed(): boolean;
	memoryBackendSession(): MemoryBackendStartOptions["session"];
	getMemoryBackend(): MemoryBackend | undefined;
	setMemoryBackend(backend: MemoryBackend): void;
	disposeMemoryBackend(): Promise<void>;
	clearMemoryBackend(): void;
	getHindsightSessionState(): HindsightSessionState | undefined;
	setHindsightSessionState(state: HindsightSessionState | undefined): void;
	getMnemopiSessionState(): MnemopiSessionState | undefined;
	takeMnemopiSessionState(): MnemopiSessionState | undefined;
	setBaseSystemPrompt(prompt: string[]): void;
	refreshBaseSystemPrompt(): Promise<void>;
	replaceMemoryTools(tools: AgentTool[]): Promise<void>;
}

/** Owns memory backend transitions and transcript-scoped memory state. */
export class SessionMemory {
	readonly #host: SessionMemoryHost;
	readonly #memoryEnabled: boolean;
	readonly #memoryAgentDir: string | undefined;
	readonly #memoryTaskDepth: number;
	readonly #createMemoryTools: (() => Promise<AgentTool[]>) | undefined;
	#memoryBackendTransition: Promise<void> = Promise.resolve();
	#localMemoryStartupAbort: AbortController | undefined;
	#baseSystemPromptBeforeMemoryPromotion: string[] | undefined;

	constructor(
		host: SessionMemoryHost,
		options: {
			memoryEnabled?: boolean;
			memoryAgentDir?: string;
			memoryTaskDepth?: number;
			createMemoryTools?: () => Promise<AgentTool[]>;
		},
	) {
		this.#host = host;
		this.#memoryAgentDir = options.memoryAgentDir;
		this.#memoryTaskDepth = options.memoryTaskDepth ?? 0;
		this.#memoryEnabled = (options.memoryEnabled ?? true) && this.#memoryTaskDepth === 0;
		this.#createMemoryTools = options.createMemoryTools;
	}

	/** Current serialized backend transition, used by prompt and disposal drains. */
	get transition(): Promise<void> {
		return this.#memoryBackendTransition;
	}

	/** Base prompt captured before a per-turn memory promotion. */
	get promotionSnapshot(): string[] | undefined {
		return this.#baseSystemPromptBeforeMemoryPromotion;
	}

	/** Clears the per-turn memory promotion after a canonical prompt rebuild. */
	clearPromotionSnapshot(): void {
		this.#baseSystemPromptBeforeMemoryPromotion = undefined;
	}

	/** Captures the canonical prompt before the first per-turn memory promotion. */
	capturePromotionSnapshot(prompt: string[]): void {
		this.#baseSystemPromptBeforeMemoryPromotion ??= prompt;
	}

	/** Restores a promotion snapshot while rolling back a failed session switch. */
	restorePromotionSnapshot(prompt: string[] | undefined): void {
		this.#baseSystemPromptBeforeMemoryPromotion = prompt;
	}
	/** Rekeys every active memory backend to the current provider session. */
	rekeyForCurrentSessionId(): void {
		this.#rekeyHindsightMemoryForCurrentSessionId();
		this.#rekeyMnemopiMemoryForCurrentSessionId();
	}

	#rekeyHindsightMemoryForCurrentSessionId(): void {
		if (this.#host.getMemoryBackend()?.id !== "hindsight") return;
		const sid = this.#host.agent.sessionId;
		if (!sid) return;
		this.#host.getHindsightSessionState()?.setSessionId(sid);
	}

	#rekeyMnemopiMemoryForCurrentSessionId(): void {
		if (this.#host.getMemoryBackend()?.id !== "mnemopi") return;
		const sid = this.#host.agent.sessionId;
		if (!sid) return;
		this.#host.getMnemopiSessionState()?.setSessionId(sid);
	}

	/** New session file: reset auto-recall / retain-threshold counters for the new transcript. */
	#resetHindsightConversationTrackingIfHindsight(): boolean {
		if (this.#host.getMemoryBackend()?.id !== "hindsight") return false;
		const state = this.#host.getHindsightSessionState();
		if (!state || state.aliasOf) return false;
		state.resetConversationTracking();
		return true;
	}

	#resetMnemopiConversationTrackingIfMnemopi(): boolean {
		if (this.#host.getMemoryBackend()?.id !== "mnemopi") return false;
		const state = this.#host.getMnemopiSessionState();
		if (!state || state.aliasOf) return false;
		state.resetConversationTracking();
		return true;
	}

	/** Applies a deferred backend selection, then resets transcript-scoped memory state. */
	async resetContextForNewTranscript(prepared?: PreparedMemoryBackendTransition): Promise<void> {
		let backend = this.#host.getMemoryBackend();
		if (this.#memoryEnabled && (prepared || (backend && backend.id !== this.#host.settings.get("memory.backend")))) {
			await this.applyMemoryBackend();
			backend = this.#host.getMemoryBackend();
		}
		this.rekeyForCurrentSessionId();
		const hadPromotedMemoryPrompt = this.#baseSystemPromptBeforeMemoryPromotion !== undefined;
		const resetHindsight = this.#resetHindsightConversationTrackingIfHindsight();
		const resetMnemopi = this.#resetMnemopiConversationTrackingIfMnemopi();
		let resetBackend = false;
		try {
			resetBackend = (await backend?.resetSession?.(this.#host.memoryBackendSession())) === true;
		} catch (error) {
			logger.warn("Memory backend reset failed", { backend: backend?.id, error: String(error) });
		}
		if (hadPromotedMemoryPrompt) {
			this.#host.setBaseSystemPrompt(this.#baseSystemPromptBeforeMemoryPromotion!);
			this.#baseSystemPromptBeforeMemoryPromotion = undefined;
		}
		if (resetHindsight || resetMnemopi || resetBackend || hadPromotedMemoryPrompt) {
			await this.#host.refreshBaseSystemPrompt();
		}
	}

	/** Cancel the local rollout-memory startup owned by this session. */
	cancelLocalMemoryStartup(): void {
		this.#localMemoryStartupAbort?.abort();
		this.#localMemoryStartupAbort = undefined;
	}

	/** Start a new local rollout-memory generation and cancel its predecessor. */
	beginLocalMemoryStartup(): AbortSignal {
		this.cancelLocalMemoryStartup();
		const controller = new AbortController();
		this.#localMemoryStartupAbort = controller;
		return controller.signal;
	}

	/** Release the local startup slot if `signal` still owns it. */
	endLocalMemoryStartup(signal: AbortSignal): void {
		if (this.#localMemoryStartupAbort?.signal === signal) this.#localMemoryStartupAbort = undefined;
	}

	async #disposeMemoryBackendState(consolidateMnemopi = true): Promise<void> {
		this.cancelLocalMemoryStartup();
		await this.#host.disposeMemoryBackend();
		this.#host.clearMemoryBackend();
		const hindsight = this.#host.getHindsightSessionState();
		if (hindsight) {
			try {
				await hindsight.flushRetainQueue();
			} catch (error) {
				logger.warn("Memory lifecycle: Hindsight flush failed", { error: String(error) });
			}
			this.#host.setHindsightSessionState(undefined);
			hindsight.dispose();
		}

		const mnemopi = this.#host.takeMnemopiSessionState();
		if (mnemopi) {
			try {
				await mnemopi.dispose({ consolidate: consolidateMnemopi });
			} catch (error) {
				logger.warn("Memory lifecycle: Mnemopi dispose failed", { error: String(error) });
			}
		}
	}

	async #runBackendTransition<T>(operation: () => Promise<T>): Promise<T> {
		const transition = this.#memoryBackendTransition.then(operation);
		this.#memoryBackendTransition = transition.then(
			() => undefined,
			() => undefined,
		);
		return await transition;
	}

	async #startMemoryBackend(backend: MemoryBackend): Promise<void> {
		this.#host.setMemoryBackend(backend);
		await backend.start({
			session: this.#host.memoryBackendSession(),
			settings: this.#host.settings,
			modelRegistry: this.#host.modelRegistry,
			agentDir: this.#memoryAgentDir!,
			taskDepth: this.#memoryTaskDepth,
		});
	}

	async #finishBackendTransition(): Promise<void> {
		if (this.#host.isDisposed()) return;
		await this.#refreshMemoryTools();
		if (this.#host.isDisposed()) return;
		await this.#host.refreshBaseSystemPrompt();
	}

	async #handleBackendTransitionError(error: unknown): Promise<never> {
		await this.#disposeMemoryBackendState(false);
		if (!this.#host.isDisposed()) {
			await this.#replaceMemoryTools([]).catch(refreshError => {
				logger.warn("Failed to remove memory tools after backend apply error", {
					error: String(refreshError),
				});
			});
		}
		throw error;
	}

	/** Dispose a changed backend while the old transcript is still active. */
	async prepareBackendForNewTranscript(): Promise<PreparedMemoryBackendTransition | undefined> {
		if (!this.#memoryEnabled || this.#host.isDisposed()) return undefined;
		return await this.#runBackendTransition(async () => {
			const previousBackend = this.#host.getMemoryBackend();
			if (!previousBackend) return undefined;
			await previousBackend.beforeTranscriptReplace?.(this.#host.memoryBackendSession());
			if (previousBackend.id === this.#host.settings.get("memory.backend")) return undefined;
			await this.#disposeMemoryBackendState();
			return { previousBackend };
		});
	}

	/** Restore the prior backend when the transcript mutation itself fails. */
	async restoreBackendAfterFailedTranscriptTransition(
		prepared: PreparedMemoryBackendTransition | undefined,
	): Promise<void> {
		if (!prepared || this.#host.isDisposed()) return;
		await this.#runBackendTransition(async () => {
			try {
				await this.#disposeMemoryBackendState(false);
				if (this.#memoryAgentDir && this.#memoryTaskDepth === 0 && !this.#host.isDisposed()) {
					await this.#startMemoryBackend(prepared.previousBackend);
				}
				await this.#finishBackendTransition();
			} catch (error) {
				await this.#handleBackendTransitionError(error);
			}
		});
	}

	/**
	 * Apply the selected memory backend to runtime state, tools, and prompt.
	 * Concurrent settings changes run in order and settle before the next turn.
	 */
	async applyMemoryBackend(): Promise<void> {
		if (!this.#memoryEnabled || this.#host.isDisposed()) return;
		await this.#runBackendTransition(() => this.#applyMemoryBackend());
	}

	async #applyMemoryBackend(): Promise<void> {
		if (this.#host.isDisposed()) return;
		try {
			await this.#disposeMemoryBackendState();
			if (this.#memoryEnabled && this.#memoryAgentDir && this.#memoryTaskDepth === 0 && !this.#host.isDisposed()) {
				await this.#startMemoryBackend(await resolveMemoryBackend(this.#host.settings));
			}
			await this.#finishBackendTransition();
		} catch (error) {
			await this.#handleBackendTransitionError(error);
		}
	}

	async #refreshMemoryTools(): Promise<void> {
		const tools = (await this.#createMemoryTools?.()) ?? [];
		await this.#replaceMemoryTools(tools);
	}

	#replaceMemoryTools(tools: AgentTool[]): Promise<void> {
		return this.#host.replaceMemoryTools(tools);
	}
}
