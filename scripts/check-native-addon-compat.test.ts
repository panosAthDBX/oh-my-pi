import { describe, expect, it } from "bun:test";
import { extractRequiredNativeBindings, missingNativeBindings } from "./check-native-addon-compat";

describe("native addon compatibility", () => {
	it("extracts the generated runtime binding surface deterministically", () => {
		const source = `
			export const DiffStream = nativeBindings.DiffStream;
			export const rasterizeSvg = nativeBindings.rasterizeSvg;
			export const duplicate = nativeBindings.DiffStream;
		`;

		expect(extractRequiredNativeBindings(source)).toEqual(["DiffStream", "rasterizeSvg"]);
	});

	it("reports only absent or undefined runtime bindings", () => {
		expect(
			missingNativeBindings(["DiffStream", "rasterizeSvg", "zeroValue"], {
				DiffStream: class {},
				rasterizeSvg: undefined,
				zeroValue: 0,
			}),
		).toEqual(["rasterizeSvg"]);
	});
});
