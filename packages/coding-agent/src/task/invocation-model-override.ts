import { createHash } from "node:crypto";

export interface TrustedTaskInvocationModelOverride {
	scopeId: string;
	toolCallId: string;
	model: string;
	agent: string;
	name: string;
	envelopeSha256: string;
}

interface StoredOverride extends TrustedTaskInvocationModelOverride {
	expiresAt: number;
}

const MAX_OVERRIDES = 256;
const OVERRIDE_TTL_MS = 10 * 60_000;
const SHA256_PATTERN = /^[a-f0-9]{64}$/;
const overrides = new Map<string, StoredOverride>();

function overrideKey(scopeId: string, toolCallId: string): string {
	return JSON.stringify([scopeId, toolCallId]);
}

function cleanExpired(now = Date.now()): void {
	for (const [key, entry] of overrides) {
		if (entry.expiresAt <= now) overrides.delete(key);
	}
}

function requireBoundedString(value: string, label: string): string {
	const trimmed = value.trim();
	if (!trimmed || trimmed.length > 256 || /[\u0000-\u001f\u007f]/.test(trimmed)) {
		throw new Error(`Invalid trusted task ${label}`);
	}
	return trimmed;
}

function stableJsonStringify(value: unknown): string {
	const normalize = (candidate: unknown): unknown => {
		if (Array.isArray(candidate)) return candidate.map(normalize);
		if (!candidate || typeof candidate !== "object") return candidate;
		const object = candidate as Record<string, unknown>;
		return Object.fromEntries(
			Object.keys(object)
				.sort()
				.map(key => [key, normalize(object[key])]),
		);
	};
	return JSON.stringify(normalize(value), null, 2);
}

function envelopeSha256(value: unknown): string {
	return createHash("sha256").update(stableJsonStringify(value)).digest("hex");
}

export function hashTrustedTaskInvocationEnvelope(value: unknown): string {
	return envelopeSha256(value);
}

/**
 * Register a one-shot model override for a trusted extension-owned task call.
 * This function is available to extension JavaScript, not the model-facing task schema.
 */
export function registerTrustedTaskInvocationModelOverride(input: TrustedTaskInvocationModelOverride): void {
	cleanExpired();
	const scopeId = requireBoundedString(input.scopeId, "scope id");
	const toolCallId = requireBoundedString(input.toolCallId, "tool call id");
	const key = overrideKey(scopeId, toolCallId);
	const digest = input.envelopeSha256.trim().toLowerCase();
	if (!SHA256_PATTERN.test(digest)) throw new Error("Invalid trusted task envelope digest");
	const normalized = {
		scopeId,
		toolCallId,
		model: requireBoundedString(input.model, "model override"),
		agent: requireBoundedString(input.agent, "agent"),
		name: requireBoundedString(input.name, "name"),
		envelopeSha256: digest,
	};
	const existing = overrides.get(key);
	if (existing) {
		if (
			existing.model === normalized.model &&
			existing.agent === normalized.agent &&
			existing.name === normalized.name &&
			existing.envelopeSha256 === normalized.envelopeSha256
		)
			return;
		throw new Error(`Conflicting trusted task model override already registered for ${toolCallId}`);
	}
	if (overrides.size >= MAX_OVERRIDES) throw new Error("Trusted task model override registry is full");
	overrides.set(key, { ...normalized, expiresAt: Date.now() + OVERRIDE_TTL_MS });
}

/** Consume and authenticate the invocation-local grant against the exact normalized task envelope. */
export function consumeTrustedTaskInvocationModelOverride(
	scopeId: string,
	toolCallId: string,
	spawn: { agent?: string; name?: string },
	spawnCount: number,
	envelope: unknown,
): string | undefined {
	cleanExpired();
	const key = overrideKey(requireBoundedString(scopeId, "scope id"), requireBoundedString(toolCallId, "tool call id"));
	const entry = overrides.get(key);
	if (!entry) return undefined;
	overrides.delete(key);
	if (
		spawnCount !== 1 ||
		spawn.agent !== entry.agent ||
		spawn.name !== entry.name ||
		envelopeSha256(envelope) !== entry.envelopeSha256
	) {
		throw new Error("Trusted task model override does not match the normalized task invocation");
	}
	return entry.model;
}

/** Verify a post-hook input revision before the agent loop accepts it. */
export function validateTrustedTaskInvocationEnvelope(
	scopeId: string,
	toolCallId: string,
	envelope: unknown,
): boolean | undefined {
	cleanExpired();
	const entry = overrides.get(overrideKey(scopeId, toolCallId));
	return entry ? envelopeSha256(envelope) === entry.envelopeSha256 : undefined;
}

export function clearTrustedTaskInvocationModelOverride(scopeId: string, toolCallId: string): void {
	overrides.delete(overrideKey(scopeId, toolCallId));
}

export function resetTrustedTaskInvocationModelOverridesForTests(): void {
	overrides.clear();
}
