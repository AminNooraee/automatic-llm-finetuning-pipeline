# Optional LiteLLM gateway registration

Gateway integration is an optional phase after verified direct serving. Missing
configuration and `gateway.enabled: false` cause zero gateway requests. The
pipeline does not install, deploy, restart, reconfigure, migrate, or directly
edit LiteLLM or its database.

## Preconditions and mutation safety

The initial path requires enabled/ready serving, `provider: litellm`,
`registration.mode: dynamic_db`, a valid gateway URL, and an API key supplied by
an exact environment reference. The provider performs a model-management
capability preflight and queries existing aliases before creating anything. If
either requested alias exists, the phase fails without overwriting or reusing it.
Authoritative conflicts come from `/model/info`; `/v1/models` is an additional
client-facing check. Credential-bearing management calls require HTTPS by default;
HTTP loopback remains available for development. An operator may deliberately set
`gateway.allow_insecure_http: true` for a trusted private/local network, but the
credential is then transmitted without TLS and a non-secret warning is emitted.
This opt-in is never automatic. Authenticated redirects are not followed.

`allow_insecure_http` controls the Project #1-to-LiteLLM transport. It is distinct
from `allow_local_backend`, which permits LiteLLM to route to a loopback/local
model-serving backend. Enabling one never enables the other.

```yaml
gateway:
  enabled: true
  provider: litellm
  base_url: http://litellm.internal:4000
  api_key: ${LITELLM_API_KEY}
  allow_insecure_http: true
```

The registrations map:

```text
gateway base alias       -> openai/<direct base alias>       -> advertised vLLM /v1 URL
gateway fine-tuned alias -> openai/<direct fine-tuned alias> -> advertised vLLM /v1 URL
```

The backend URL comes from `serving.advertise_host` and `serving.port`; it is
never forced to localhost. Each creation request carries the current pipeline
run ID as ownership evidence. No delete/update method is implemented. If the
first creation succeeds and the second fails, the registration artifact records
`partial_failed`, the known current-run object is preserved, and the operator
receives a non-secret diagnostic. A failed first creation is recorded as
`registration_failed`, not as a partial registration.

Every mutation attempt is recorded before it is sent. A timeout or non-success
response is reconciled against `/model/info` using the alias, role, and current
run ID, then classified as created, absent, conflicting, or indeterminate.
Indeterminate requests are never retried automatically. LiteLLM receives the non-secret
`not-required` placeholder as the vLLM backend API key; it is separate from the
gateway management credential.

## Verification and handoff

After both creations, the manager requires both aliases in LiteLLM `/v1/models`
and performs a real non-streaming Chat Completions request through LiteLLM for
each alias. This verifies the complete LiteLLM → vLLM → base/LoRA chains.

```text
gateway/
  gateway_manifest.json
  registration.json
  health_check.json
  gateway.log
```

The gateway manifest becomes the preferred `output.endpoint_handoff`. The
registration artifact contains aliases, non-secret backend mappings, optional
registration IDs, status, and ownership evidence. No authorization header or
credential is persisted.

## Secrets and validation status

Resolved API keys live only in memory. Snapshot and JSON writers recursively
redact sensitive keys, run log handlers redact registered secret values, and
provider/verifier exceptions redact values echoed by upstream systems. Tests use
fake secrets and scan run artifacts.

The dynamic `/model/info` + `/model/new` shape, conflict/auth/capability/partial
failure behavior, both inference checks, and artifacts are mock-validated. They
have not yet been accepted against a real LiteLLM persistent-database deployment;
operators must qualify the exact LiteLLM version and its enabled management API
in an isolated environment before production use.
