import { describe, expect, it } from "vitest";
import { Settings, type ExtensionContext } from "@oh-my-pi/pi-coding-agent";
import { resolveModelRoleValue } from "@oh-my-pi/pi-coding-agent/config/model-resolver";
import { resolveOmpRole } from "../src/core/omp-model-role.js";
import { validateNativeThinking } from "../src/core/model-resolution.js";

const models = [
  { provider: "test", id: "second", name: "Second", reasoning: true },
] as unknown as ReturnType<ExtensionContext["modelRegistry"]["getAvailable"]>;

function fixture(roles: Record<string, string>, registry = models) {
  const settings = Settings.isolated({ modelRoles: roles });
  const context = {
    modelRegistry: { getAvailable: () => registry },
    models: { resolve: (selector: string) => resolveModelRoleValue(selector, registry, { settings }).model },
  } as unknown as ExtensionContext;
  return { settings, context };
}

describe("OMP role selection", () => {
  it("selects the first available configured candidate with its thinking", () => {
    const { settings, context } = fixture({ first: "test/missing:low", second: "test/second:high" });
    expect(resolveOmpRole("@first, @second", context, settings)).toEqual({ model: "test/second", thinking: "high" });
  });

  it("rejects an unavailable role instead of choosing a fuzzy Fabric model", () => {
    const { settings, context } = fixture({ missing: "test/not-present" });
    expect(() => resolveOmpRole("@missing", context, settings)).toThrow("did not resolve against this session's model roles");
  });

  it.each(["auto", "inherit"])("retains native %s thinking on the canonical model", (thinking) => {
    const { settings, context } = fixture({ fast: `test/second:${thinking}` });
    expect(resolveOmpRole("@fast", context, settings)).toEqual({ model: "test/second", nativeThinking: thinking });
  });

  it.each(["auto", "inherit"] as const)("rejects case-insensitive native %s collisions", (mode) => {
    const available = [{ provider: "TEST", id: `SECOND:${mode.toUpperCase()}` }];
    expect(() => validateNativeThinking("test/second", mode, available)).toThrow(
      `OMP role native thinking selector conflicts with a literal model ID: test/second:${mode}`,
    );
    expect(() => validateNativeThinking("test/second", undefined, available)).not.toThrow();
  });

  it("uses the matched fallback candidate native mode", () => {
    const { settings, context } = fixture({ first: "test/missing:high", second: "test/second:inherit" });
    expect(resolveOmpRole("@first, @second", context, settings)).toEqual({ model: "test/second", nativeThinking: "inherit" });
  });

  it("allows an explicit effort to override an automatic role effort", () => {
    const { settings, context } = fixture({ fast: "test/second:auto" });
    expect(resolveOmpRole("@fast", context, settings, true)).toEqual({ model: "test/second" });
  });

  it.each(["auto", "inherit"])("treats a literal %s-suffixed registry ID as the model, not a thinking suffix", (suffix) => {
    const registry = [
      { provider: "test", id: "second", name: "Second", reasoning: true },
      { provider: "test", id: `second:${suffix}`, name: "Second " + suffix, reasoning: true },
    ] as unknown as ReturnType<ExtensionContext["modelRegistry"]["getAvailable"]>;
    const { settings, context } = fixture({ fast: `test/second:${suffix}` }, registry);
    expect(resolveOmpRole("@fast", context, settings)).toEqual({ model: `test/second:${suffix}` });
  });
});
