import { createRequire } from "node:module";

const NATIVE_BINDING_RE = /\bnativeBindings\.([A-Za-z_$][\w$]*)/g;

export function extractRequiredNativeBindings(indexSource: string): string[] {
	return [...new Set([...indexSource.matchAll(NATIVE_BINDING_RE)].map(match => match[1]!))].sort();
}

export function missingNativeBindings(
	required: readonly string[],
	bindings: Readonly<Record<string, unknown>>,
): string[] {
	return required.filter(name => bindings[name] === undefined);
}

if (import.meta.main) {
	const [indexPath, ...addonPaths] = process.argv.slice(2);
	if (!indexPath || addonPaths.length === 0) {
		throw new Error("Usage: bun scripts/check-native-addon-compat.ts <native-index.js> <addon.node>...");
	}
	const required = extractRequiredNativeBindings(await Bun.file(indexPath).text());
	if (required.length === 0) throw new Error(`No native binding references found in ${indexPath}`);

	const require = createRequire(import.meta.url);
	let compatible = true;
	for (const addonPath of addonPaths) {
		const bindings = require(addonPath) as Record<string, unknown>;
		if (Object.keys(bindings).length === 0) throw new Error(`Native addon exported no bindings: ${addonPath}`);
		const missing = missingNativeBindings(required, bindings);
		if (missing.length === 0) continue;
		compatible = false;
		console.error(`Native addon is older than the checkout (${addonPath}): missing ${missing.join(", ")}`);
	}
	if (!compatible) process.exitCode = 2;
}
