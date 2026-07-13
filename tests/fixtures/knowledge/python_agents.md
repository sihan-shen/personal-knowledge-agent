# Python AI Agents

## Tools 工具

Agent 工具具有明确的输入输出契约。An Agent calls a tool when external information or an action is required, and records the observation in its state.

## State and Decisions

An Agent maintains state across steps. Its decision policy chooses the next tool or decides to return a final answer based on the current goal and observations.

## Safety

工具调用前应验证参数，调用后应检查结果。Failures should remain visible instead of being silently converted into successful observations.
