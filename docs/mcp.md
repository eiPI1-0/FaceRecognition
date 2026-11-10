# MCP tools for LLM clients

Four read-only tools, each with a JSON Schema, served in-process — and one deliberate absence.

Deep dive split out of [README.md](../README.md), which keeps the short version.
Related: [serving.md](serving.md) (the HTTP transport shares its access control), [interfaces.md](interfaces.md)

---

### mcp — Model Context Protocol tools

```sh
./main mcp                        # stdio (Claude Desktop / IDE)
./main mcp --transport http --port 8081 --workers 8    # POST /mcp
```

Four read-only tools, each with a JSON Schema:

| Tool | Arguments | Returns |
|---|---|---|
| `identify_faces` | `image_b64`, `top_k`, `min_score`, `include_landmarks` | per-face bbox (original pixels), `matched` or `null`, ranked candidates |
| `compare_faces` | `image_a_b64`, `image_b_b64` | cosine of the most confident face in each (identify's `faces[0]`) |
| `list_people` | — | enrolled person ids (never embeddings or photos) |
| `get_face_crop` | `image_b64`, `face_index` | aligned 112×112 PNG as `image` content |

**Enrollment is deliberately not a tool.** Writing biometric templates is a
non-retractable action and must not be reachable from a prompt-injectable caller; use
`facerec enroll` or an admin API instead. Calling `enroll_person` returns
`-32602 unknown tool`.

The `identify_faces` description (which the model actually reads) states that
`"matched": null` means *unknown*, and that candidates must not be turned into an
identity — a low score is not a weak guess, it is "not this person".

Claude Desktop / MCP client config:

```json
{"mcpServers":{"facerec":{"command":"/abs/path/build/main",
                          "args":["mcp"],
                          "cwd":"/abs/path/build"}}}
```

`mcp` links the library directly rather than shelling out, so a tool call costs one
forward pass instead of a cold `readNet` of both ONNX files.
