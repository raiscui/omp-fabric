import type { ExtensionContext } from "@oh-my-pi/pi-coding-agent";
import { extractExplicitThinkingSelector, resolveConfiguredModelPatterns, resolveModelRoleValue } from "@oh-my-pi/pi-coding-agent/config/model-resolver";
import type { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { isFabricThinking, type FabricThinking } from "../thinking.js";
import { validateNativeThinking } from "./model-resolution.js";

export type OmpRoleSelection = { model: string; thinking?: FabricThinking; nativeThinking?: "auto" | "inherit" };

export function resolveOmpRole(selector: string, context: ExtensionContext, settings: Settings, explicitThinking = false): OmpRoleSelection {
  const available = context.modelRegistry.getAvailable();
  const resolved = resolveModelRoleValue(selector, available, { settings });
  if (!resolved.model) {
    throw new Error(
      "OMP role selector " + JSON.stringify(selector) +
      " did not resolve against this session's model roles and authenticated models; a detached Fabric host cannot resolve a raw @role without an authoritative session",
    );
  }
  const hostModel = context.models.resolve(selector);
  if (!hostModel || resolved.model.provider !== hostModel.provider || resolved.model.id !== hostModel.id) {
    throw new Error("OMP role selector " + JSON.stringify(selector) + " could not be resolved consistently in this session");
  }
  const patterns = resolveConfiguredModelPatterns(selector, settings);
  const matched = resolved.matchedPatternIndex === undefined ? undefined : patterns?.[resolved.matchedPatternIndex];
  const explicit = resolved.explicitThinkingLevel ? extractExplicitThinkingSelector(matched, settings, {
    isLiteralModelId: (provider, id) => context.modelRegistry.getAvailable().some((model) => model.id === id && (provider === undefined || model.provider === provider)),
  }) : undefined;
  const nativeThinking = !explicitThinking && (explicit === "auto" || explicit === "inherit") ? explicit : undefined;
  validateNativeThinking(`${resolved.model.provider}/${resolved.model.id}`, nativeThinking, available);
  return { model: resolved.model.provider + "/" + resolved.model.id, ...(nativeThinking ? { nativeThinking } : {}), ...(!explicitThinking && !nativeThinking && isFabricThinking(resolved.thinkingLevel) ? { thinking: resolved.thinkingLevel } : {}) };
}
