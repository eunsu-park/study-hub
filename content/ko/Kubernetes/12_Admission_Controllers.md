# 12. 어드미션 컨트롤러(Admission Controllers)

**이전**: [오퍼레이터](./11_Operators.md) | **다음**: [오토스케일링](./13_Autoscaling.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Kubernetes 어드미션 컨트롤러 파이프라인(admission controller pipeline)과 요청 흐름을 설명할 수 있다
2. 검증(validating) 및 변이(mutating) 웹훅 서버를 구성하고 배포할 수 있다
3. OPA Gatekeeper 제약 조건(constraint)과 제약 조건 템플릿(constraint template)을 사용한 정책 시행을 구현할 수 있다
4. 선언적 YAML 정책을 사용하는 대안 정책 엔진으로 Kyverno를 사용할 수 있다
5. 개발 및 프로덕션 환경에서 어드미션 정책을 테스트하고 디버깅할 수 있다

---

Kubernetes API 서버로 들어오는 모든 요청은 etcd에 저장되기 전에 어드미션 컨트롤러(admission controller) 체인을 통과합니다. 이 체인은 Kubernetes에서 가장 강력한 확장 지점 중 하나입니다 -- 보안 정책 적용, 사이드카 주입, 기본값 설정, 구성 유효성 검증, 잘못된 구성이 클러스터에 도달하기 전에 방지할 수 있습니다. 이 레슨에서는 내장 어드미션 컨트롤러와 사용자 정의 로직을 연결할 수 있는 동적 어드미션 제어(dynamic admission control) 시스템을 모두 다룹니다.

## 목차

- [이론과 원리](#이론과-원리)
- [1. 어드미션 컨트롤러 파이프라인](#1-the-admission-controller-pipeline)
- [2. 내장 어드미션 컨트롤러](#2-built-in-admission-controllers)
- [3. 동적 어드미션 제어](#3-dynamic-admission-control)
- [4. 검증 웹훅](#4-validating-webhooks)
- [5. 변이 웹훅](#5-mutating-webhooks)
- [6. 웹훅 구성](#6-webhook-configuration)
- [7. OPA Gatekeeper](#7-opa-gatekeeper)
- [8. Kyverno](#8-kyverno)
- [9. 어드미션 정책 테스팅](#9-testing-admission-policies)
- [10. 어드미션 컨트롤러 성능](#10-admission-controller-performance)
- [연습문제](#exercises)

---

## 1. 어드미션 컨트롤러 파이프라인

### 이론: 어드미션이 어디에 있고 왜 중요한가

6강의 4-게이트 파이프라인을 떠올려보세요 — **authn → authz → admission → schema validation → etcd 영속화**. 각 단계는 특정 질문에 답합니다:

- **Authn** — 당신은 누구인가?
- **Authz** (RBAC 등) — 이 리소스에 이 verb를 허용받았는가?
- **Admission** — 이 *특정* 요청이 통과되어야 하는가?
- **Schema/CEL validation** — 객체가 등록된 형태와 일치하는가?
- **Persist** — etcd에 쓰기.

어드미션은 권한이 아니라 객체의 *내용*에 의존하는 모든 것에 대한 *바로 그* 확장 지점입니다. RBAC는 "create pods"를 부여할 수 있지만 "non-privileged 파드만"이라 말할 수 없습니다. 스키마 검증은 필드를 요구할 수 있지만 "이 필드의 값이 우리 레지스트리의 호스트명과 일치"를 강제할 수 없습니다. 두 격차 모두 어드미션의 일입니다.

어드미션은 또한 **영속화 전에** 실행되므로, 거부된 요청은 etcd에 들어가지 않고, 부분 상태에 대한 audit 로그 노이즈를 만들지 않으며, 컨트롤러를 혼란시키지 않습니다. 이것이 "어드미션으로서의 정책"이 "사후에 나쁜 객체를 삭제하는 컨트롤러로서의 정책"과 근본적으로 다른 이유입니다 — 후자는 나쁜 상태가 존재하고 다른 컨트롤러가 관찰할 수 있는 창을 만들고, 전자는 나쁜 상태를 문자 그대로 만들 수 없게 합니다.

트레이드오프 — 어드미션은 핫 패스에 있습니다. 모든 API 요청 — 모든 kubectl apply, 모든 컨트롤러 create — 이 어드미션 비용을 지불합니다. 그래서 웹훅은 엄격한 성능 요구사항을 가집니다(§C).

### 1.1 요청 흐름

클라이언트(kubectl, 컨트롤러, CI 파이프라인)가 API 서버에 요청을 보내면 여러 단계를 거칩니다:

```
Client Request
    │
    ▼
┌──────────────────┐
│  Authentication   │  Who are you?
└────────┬─────────┘
         │
    ▼
┌──────────────────┐
│  Authorization    │  Are you allowed?
│  (RBAC)          │
└────────┬─────────┘
         │
    ▼
┌──────────────────────────────────────────────┐
│           Admission Controllers               │
│                                               │
│  ┌─────────────────┐  ┌────────────────────┐  │
│  │    Mutating      │  │    Validating      │  │
│  │    Admission     │──▶    Admission       │  │
│  │    Webhooks      │  │    Webhooks        │  │
│  └─────────────────┘  └────────────────────┘  │
│                                               │
│  (Object schema validation happens between)   │
└────────┬─────────────────────────────────────┘
         │
    ▼
┌──────────────────┐
│  Persist to etcd  │
└──────────────────┘
```

### 1.2 변이(Mutating) vs 검증(Validating)

| 단계 | 목적 | 객체 수정 가능? | 실행 순서 |
|---|---|---|---|
| 변이(Mutating) | 기본값 설정, 사이드카 주입, 레이블 추가 | 예 | 먼저 |
| 객체 스키마 유효성 검증 | OpenAPI 스키마에 대해 확인 | 아니오 | 중간 |
| 검증(Validating) | 정책 시행, 잘못된 구성 거부 | 아니오 | 나중에 |

검증기가 객체의 최종 형태를 보아야 하므로 변이 웹훅이 먼저 실행됩니다. 변이 웹훅은 한 변이기의 변경이 재평가를 트리거하는 경우 여러 번 호출될 수도 있습니다.

---

## 2. 내장 어드미션 컨트롤러

### 이론: 내장 어드미션 플러그인과 2-패스 웹훅 설계

어드미션은 두 종류의 플러그인을 가집니다 — **내장**(API 서버에 컴파일됨)과 **동적**(등록하는 웹훅). 내장 플러그인은 보편적 케이스를 처리합니다:

- `LimitRanger`는 네임스페이스에 `LimitRange`가 있으면 기본 CPU/메모리 limit을 주입.
- `ResourceQuota`는 네임스페이스의 `ResourceQuota`를 초과할 요청을 거부.
- `ServiceAccount`는 네임스페이스의 default ServiceAccount와 그 토큰 볼륨을 주입.
- `NamespaceLifecycle`은 존재하지 않거나 종료 중인 네임스페이스에서의 create를 거부.
- `PodSecurity` (6강)는 Pod Security Standards 강제.
- `MutatingAdmissionWebhook`과 `ValidatingAdmissionWebhook`은 동적 어드미션의 진입점.

두 웹훅 유형은 별개의 단계에서 실행됩니다:

**Phase 1 — Mutating 웹훅.** 요청에 매치되는 등록된 각 `MutatingWebhookConfiguration`이 객체와 함께 호출됩니다. 각각은 API 서버가 적용할 JSON patch를 반환할 수 있습니다. Mutating 웹훅은 체이닝됩니다 — 웹훅 A의 출력이 웹훅 B의 입력 — 그래서 순서가 중요할 수 있습니다(API 서버는 비결정적 순서로 처리하며, 후속 mutation 후 재실행을 위한 `reinvocationPolicy: IfNeeded`). 전형적 사용 — 사이드카 주입(Istio, Linkerd), 레이블/어노테이션 추가, API 작성자가 잊은 기본값 설정.

**Phase 2 — Validating 웹훅.** 모든 mutation이 끝난 후, validating 웹훅이 *최종* 객체를 보고 allow나 deny를 반환(patch 없음). 여러 validating 웹훅이 모두 실행됩니다 — 어느 하나가 거부하면 요청이 거부됩니다. 전형적 사용 — 정책 강제(privileged 파드 금지, 이미지 레지스트리, 레이블 요구사항).

이 순서는 의도적입니다 — 검증은 최종 상태에서 실행되므로, mutator가 기본값을 추가하고 validator가 그것을 검증할 수 있고, 사용자는 최종 형태에 대한 한 번의 오류 메시지만 봅니다. 다른 순서로 하면 validator가 부분 객체를 승인했다가 후속 mutator가 깨뜨릴 수 있습니다.

미묘한 점 — mutating 웹훅은 멱등성과 충돌에 주의해야 합니다. 두 웹훅이 같은 어노테이션을 다른 값으로 주입하려 하면, API 서버의 reinvocation 로직이 해결하지만 사용자는 예측 불가 동작을 얻습니다. 운영 규칙은 — 각 mutating 웹훅은 겹치지 않는 관심사를 소유해야 합니다.

### 2.1 일반적으로 사용되는 컨트롤러

Kubernetes는 약 30개의 컴파일된 어드미션 컨트롤러를 제공합니다. 주요 컨트롤러:

| 컨트롤러 | 목적 |
|---|---|
| `NamespaceLifecycle` | 존재하지 않거나 종료 중인 네임스페이스에서의 작업 방지 |
| `LimitRanger` | Pod에 대한 LimitRange 제약 조건 적용 |
| `ServiceAccount` | 기본 서비스 계정(service account)과 토큰 주입 |
| `DefaultStorageClass` | 클래스가 없는 PVC에 기본 StorageClass 설정 |
| `ResourceQuota` | 네임스페이스 리소스 할당량(quota) 적용 |
| `PodSecurity` | Pod Security Standards 적용 (PodSecurityPolicy 대체) |
| `MutatingAdmissionWebhook` | 외부 변이 웹훅 호출 |
| `ValidatingAdmissionWebhook` | 외부 검증 웹훅 호출 |
| `ValidatingAdmissionPolicy` | 클러스터 내 CEL 기반 유효성 검증 (v1.28+ stable) |

### 2.2 활성화된 컨트롤러 확인

```bash
# Check which admission plugins are enabled
kubectl exec -n kube-system kube-apiserver-<node> -- \
  kube-apiserver --help 2>&1 | grep enable-admission-plugins

# Or check the API server manifest
cat /etc/kubernetes/manifests/kube-apiserver.yaml | grep admission

# Default enabled list (Kubernetes 1.29+):
# CertificateApproval, CertificateSigning, CertificateSubjectRestriction,
# DefaultIngressClass, DefaultStorageClass, DefaultTolerationSeconds,
# LimitRanger, MutatingAdmissionWebhook, NamespaceLifecycle,
# PersistentVolumeClaimResize, PodSecurity, Priority,
# ResourceQuota, RuntimeClass, ServiceAccount, StorageObjectInUseProtection,
# TaintNodesByCondition, ValidatingAdmissionPolicy, ValidatingAdmissionWebhook
```

### 2.3 ValidatingAdmissionPolicy (CEL 기반)

Kubernetes 1.28+는 외부 웹훅 없이 Common Expression Language(CEL)를 사용한 클러스터 내 유효성 검증을 제공합니다:

```yaml
# Define a policy
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicy
metadata:
  name: require-resource-limits
spec:
  failurePolicy: Fail
  matchConstraints:
    resourceRules:
    - apiGroups: [""]
      apiVersions: ["v1"]
      operations: ["CREATE", "UPDATE"]
      resources: ["pods"]
  validations:
  - expression: >-
      object.spec.containers.all(c,
        has(c.resources) &&
        has(c.resources.limits) &&
        has(c.resources.limits.cpu) &&
        has(c.resources.limits.memory)
      )
    message: "All containers must have CPU and memory limits set"
    reason: Invalid
---
# Bind the policy to a namespace
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicyBinding
metadata:
  name: require-resource-limits-binding
spec:
  policyName: require-resource-limits
  validationActions:
  - Deny
  matchResources:
    namespaceSelector:
      matchLabels:
        environment: production
```

---

## 3. 동적 어드미션 제어

### 3.1 아키텍처

동적 어드미션 제어(dynamic admission control)를 사용하면 API 서버가 어드미션 단계에서 호출하는 외부 HTTPS 서버(웹훅)를 등록할 수 있습니다.

```
                  API Server
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
  ┌───────────────┐   ┌────────────────┐
  │   Mutating    │   │  Validating    │
  │   Webhook     │   │  Webhook       │
  │   Config      │   │  Config        │
  └───────┬───────┘   └────────┬───────┘
          │                     │
          ▼                     ▼
  ┌───────────────┐   ┌────────────────┐
  │  Webhook Pod  │   │  Webhook Pod   │
  │  (HTTPS)      │   │  (HTTPS)       │
  │  /mutate      │   │  /validate     │
  └───────────────┘   └────────────────┘
```

### 3.2 AdmissionReview API

웹훅은 `AdmissionReview` 객체를 사용하여 통신합니다:

```json
{
  "apiVersion": "admission.k8s.io/v1",
  "kind": "AdmissionReview",
  "request": {
    "uid": "705ab4f5-6393-11e8-b7cc-42010a800002",
    "kind": {"group": "", "version": "v1", "kind": "Pod"},
    "resource": {"group": "", "version": "v1", "resource": "pods"},
    "namespace": "default",
    "operation": "CREATE",
    "userInfo": {
      "username": "system:serviceaccount:default:my-sa",
      "groups": ["system:serviceaccounts"]
    },
    "object": {
      "metadata": {"name": "my-pod", "namespace": "default"},
      "spec": {
        "containers": [{"name": "app", "image": "nginx:latest"}]
      }
    },
    "oldObject": null,
    "dryRun": false
  }
}
```

---

## 4. 검증 웹훅

### 4.1 웹훅 서버 구현 (Go)

```go
package main

import (
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "strings"

    admissionv1 "k8s.io/api/admission/v1"
    corev1 "k8s.io/api/core/v1"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/apimachinery/pkg/runtime"
    "k8s.io/apimachinery/pkg/runtime/serializer"
)

var (
    runtimeScheme = runtime.NewScheme()
    codecs        = serializer.NewCodecFactory(runtimeScheme)
    deserializer  = codecs.UniversalDeserializer()
)

// validatePod checks that no container uses the :latest tag
func validatePod(pod *corev1.Pod) (bool, string) {
    for _, container := range pod.Spec.Containers {
        if strings.HasSuffix(container.Image, ":latest") || !strings.Contains(container.Image, ":") {
            return false, fmt.Sprintf(
                "container %q uses image %q: images must have an explicit tag, not :latest",
                container.Name, container.Image,
            )
        }
    }
    for _, container := range pod.Spec.InitContainers {
        if strings.HasSuffix(container.Image, ":latest") || !strings.Contains(container.Image, ":") {
            return false, fmt.Sprintf(
                "init container %q uses image %q: images must have an explicit tag, not :latest",
                container.Name, container.Image,
            )
        }
    }
    return true, ""
}

func handleValidate(w http.ResponseWriter, r *http.Request) {
    body, err := io.ReadAll(r.Body)
    if err != nil {
        http.Error(w, "could not read body", http.StatusBadRequest)
        return
    }

    var admissionReview admissionv1.AdmissionReview
    if _, _, err := deserializer.Decode(body, nil, &admissionReview); err != nil {
        http.Error(w, "could not decode body", http.StatusBadRequest)
        return
    }

    var pod corev1.Pod
    if err := json.Unmarshal(admissionReview.Request.Object.Raw, &pod); err != nil {
        http.Error(w, "could not unmarshal pod", http.StatusBadRequest)
        return
    }

    allowed, reason := validatePod(&pod)

    response := &admissionv1.AdmissionReview{
        TypeMeta: metav1.TypeMeta{
            APIVersion: "admission.k8s.io/v1",
            Kind:       "AdmissionReview",
        },
        Response: &admissionv1.AdmissionResponse{
            UID:     admissionReview.Request.UID,
            Allowed: allowed,
        },
    }
    if !allowed {
        response.Response.Result = &metav1.Status{
            Message: reason,
            Code:    http.StatusForbidden,
        }
    }

    respBytes, _ := json.Marshal(response)
    w.Header().Set("Content-Type", "application/json")
    w.Write(respBytes)
}

func main() {
    http.HandleFunc("/validate", handleValidate)
    fmt.Println("Starting webhook server on :8443")
    err := http.ListenAndServeTLS(":8443", "/certs/tls.crt", "/certs/tls.key", nil)
    if err != nil {
        panic(err)
    }
}
```

### 4.2 웹훅 서버 배포

```yaml
# Deployment for the webhook server
apiVersion: apps/v1
kind: Deployment
metadata:
  name: image-policy-webhook
  namespace: webhook-system
spec:
  replicas: 2
  selector:
    matchLabels:
      app: image-policy-webhook
  template:
    metadata:
      labels:
        app: image-policy-webhook
    spec:
      containers:
      - name: webhook
        image: example.com/image-policy-webhook:v1.0.0
        ports:
        - containerPort: 8443
          protocol: TCP
        volumeMounts:
        - name: tls-certs
          mountPath: /certs
          readOnly: true
        resources:
          requests:
            cpu: 50m
            memory: 64Mi
          limits:
            cpu: 200m
            memory: 128Mi
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8443
            scheme: HTTPS
          initialDelaySeconds: 5
        readinessProbe:
          httpGet:
            path: /readyz
            port: 8443
            scheme: HTTPS
          initialDelaySeconds: 3
      volumes:
      - name: tls-certs
        secret:
          secretName: image-policy-webhook-tls
---
apiVersion: v1
kind: Service
metadata:
  name: image-policy-webhook
  namespace: webhook-system
spec:
  selector:
    app: image-policy-webhook
  ports:
  - port: 443
    targetPort: 8443
    protocol: TCP
```

### 4.3 TLS 인증서 설정

```bash
# Generate CA and server certificate using OpenSSL
# The SAN must match the webhook service DNS name
SERVICE_NAME=image-policy-webhook
NAMESPACE=webhook-system

# Generate CA key and certificate
openssl genrsa -out ca.key 2048
openssl req -new -x509 -days 365 -key ca.key -subj "/CN=Webhook CA" -out ca.crt

# Generate server key and CSR
openssl genrsa -out server.key 2048
openssl req -new -key server.key \
  -subj "/CN=${SERVICE_NAME}.${NAMESPACE}.svc" \
  -out server.csr

# Create SAN config
cat > san.cnf <<EOF
[req]
req_extensions = v3_req
[v3_req]
subjectAltName = DNS:${SERVICE_NAME}.${NAMESPACE}.svc, DNS:${SERVICE_NAME}.${NAMESPACE}.svc.cluster.local
EOF

# Sign the server certificate
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key \
  -CAcreateserial -out server.crt -days 365 \
  -extfile san.cnf -extensions v3_req

# Create the TLS secret
kubectl create secret tls image-policy-webhook-tls \
  --cert=server.crt --key=server.key \
  -n webhook-system

# The CA bundle (base64-encoded) goes in the webhook configuration
CA_BUNDLE=$(cat ca.crt | base64 | tr -d '\n')
echo $CA_BUNDLE
```

또는 자동 인증서 관리를 위해 cert-manager를 사용할 수 있습니다:

```yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: image-policy-webhook-cert
  namespace: webhook-system
spec:
  secretName: image-policy-webhook-tls
  dnsNames:
  - image-policy-webhook.webhook-system.svc
  - image-policy-webhook.webhook-system.svc.cluster.local
  issuerRef:
    name: selfsigned-issuer
    kind: ClusterIssuer
  duration: 8760h  # 1 year
  renewBefore: 720h  # 30 days
```

---

## 5. 변이 웹훅

### 5.1 JSON Patch를 통한 변이

변이 웹훅(mutating webhook)은 들어오는 객체를 수정하기 위해 응답에 JSON Patch(RFC 6902)를 반환합니다:

```go
// handleMutate injects a sidecar container into every pod
func handleMutate(w http.ResponseWriter, r *http.Request) {
    body, _ := io.ReadAll(r.Body)
    var admissionReview admissionv1.AdmissionReview
    deserializer.Decode(body, nil, &admissionReview)

    var pod corev1.Pod
    json.Unmarshal(admissionReview.Request.Object.Raw, &pod)

    // Skip if annotation says no injection
    if pod.Annotations["sidecar-injector/inject"] == "false" {
        sendAllowed(w, admissionReview.Request.UID)
        return
    }

    // Build JSON Patch to add a sidecar container
    patches := []map[string]interface{}{}

    sidecar := map[string]interface{}{
        "name":  "log-collector",
        "image": "fluent/fluent-bit:2.2",
        "resources": map[string]interface{}{
            "requests": map[string]string{"cpu": "25m", "memory": "32Mi"},
            "limits":   map[string]string{"cpu": "100m", "memory": "64Mi"},
        },
        "volumeMounts": []map[string]string{
            {"name": "shared-logs", "mountPath": "/var/log/app"},
        },
    }

    patches = append(patches, map[string]interface{}{
        "op":    "add",
        "path":  "/spec/containers/-",
        "value": sidecar,
    })

    // Add shared volume if no volumes exist
    if len(pod.Spec.Volumes) == 0 {
        patches = append(patches, map[string]interface{}{
            "op":   "add",
            "path": "/spec/volumes",
            "value": []map[string]interface{}{
                {"name": "shared-logs", "emptyDir": map[string]interface{}{}},
            },
        })
    } else {
        patches = append(patches, map[string]interface{}{
            "op":   "add",
            "path": "/spec/volumes/-",
            "value": map[string]interface{}{
                "name": "shared-logs", "emptyDir": map[string]interface{}{},
            },
        })
    }

    // Add a label to track injection
    if pod.Labels == nil {
        patches = append(patches, map[string]interface{}{
            "op":    "add",
            "path":  "/metadata/labels",
            "value": map[string]string{"sidecar-injected": "true"},
        })
    } else {
        patches = append(patches, map[string]interface{}{
            "op":    "add",
            "path":  "/metadata/labels/sidecar-injected",
            "value": "true",
        })
    }

    patchBytes, _ := json.Marshal(patches)
    patchType := admissionv1.PatchTypeJSONPatch

    response := &admissionv1.AdmissionReview{
        TypeMeta: metav1.TypeMeta{
            APIVersion: "admission.k8s.io/v1",
            Kind:       "AdmissionReview",
        },
        Response: &admissionv1.AdmissionResponse{
            UID:       admissionReview.Request.UID,
            Allowed:   true,
            Patch:     patchBytes,
            PatchType: &patchType,
        },
    }

    respBytes, _ := json.Marshal(response)
    w.Header().Set("Content-Type", "application/json")
    w.Write(respBytes)
}
```

### 5.2 일반적인 변이 사용 사례

| 사용 사례 | 변이되는 것 |
|---|---|
| 사이드카 주입 (Istio, Linkerd) | 컨테이너와 볼륨 추가 |
| 기본 리소스 제한 | 누락된 경우 request/limit 설정 |
| 이미지 레지스트리 재작성 | `nginx`를 `registry.internal/nginx`로 교체 |
| 레이블/어노테이션 주입 | 조직 표준 레이블 추가 |
| 노드 어피니티 주입 | 네임스페이스 기반 toleration 또는 nodeSelector 추가 |
| 환경 변수 주입 | 공통 env 변수(region, cluster name) 추가 |

---

## 6. 웹훅 구성

### 6.1 ValidatingWebhookConfiguration

```yaml
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
metadata:
  name: image-policy-validator
  annotations:
    cert-manager.io/inject-ca-from: webhook-system/image-policy-webhook-cert
webhooks:
- name: validate.image-policy.example.com
  admissionReviewVersions: ["v1"]
  sideEffects: None
  timeoutSeconds: 5
  failurePolicy: Fail      # Fail or Ignore
  matchPolicy: Equivalent   # Exact or Equivalent
  clientConfig:
    service:
      name: image-policy-webhook
      namespace: webhook-system
      path: /validate
      port: 443
    # caBundle: <base64-encoded CA cert>  # Use if not using cert-manager annotation
  rules:
  - operations: ["CREATE", "UPDATE"]
    apiGroups: [""]
    apiVersions: ["v1"]
    resources: ["pods"]
    scope: "Namespaced"
  namespaceSelector:
    matchExpressions:
    - key: kubernetes.io/metadata.name
      operator: NotIn
      values: ["kube-system", "kube-public", "webhook-system"]
  objectSelector:
    matchExpressions:
    - key: skip-validation
      operator: DoesNotExist
```

### 6.2 MutatingWebhookConfiguration

```yaml
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingWebhookConfiguration
metadata:
  name: sidecar-injector
webhooks:
- name: mutate.sidecar-injector.example.com
  admissionReviewVersions: ["v1"]
  sideEffects: None
  reinvocationPolicy: IfNeeded  # Re-invoke if another mutator changes the object
  timeoutSeconds: 10
  failurePolicy: Ignore          # Do not block pod creation if webhook is down
  clientConfig:
    service:
      name: sidecar-injector
      namespace: webhook-system
      path: /mutate
      port: 443
    caBundle: "${CA_BUNDLE}"
  rules:
  - operations: ["CREATE"]
    apiGroups: [""]
    apiVersions: ["v1"]
    resources: ["pods"]
  namespaceSelector:
    matchLabels:
      sidecar-injection: enabled
```

### 6.3 구성 필드 참조

| 필드 | 설명 | 옵션 |
|---|---|---|
| `failurePolicy` | 웹훅에 연결할 수 없을 때 수행할 작업 | `Fail` (거부) 또는 `Ignore` (허용) |
| `sideEffects` | 웹훅에 부작용이 있는지 여부 | `None`, `NoneOnDryRun`, `Unknown` |
| `timeoutSeconds` | 웹훅 응답을 기다리는 최대 시간 | 1-30 (기본값: 10) |
| `reinvocationPolicy` | 다른 변이 후 재호출 여부 | `Never` 또는 `IfNeeded` |
| `matchPolicy` | API 버전 매칭 방법 | `Exact` 또는 `Equivalent` |
| `namespaceSelector` | 네임스페이스를 필터링하는 레이블 셀렉터 | 표준 레이블 셀렉터 |
| `objectSelector` | 객체를 필터링하는 레이블 셀렉터 | 표준 레이블 셀렉터 |

---

## 7. OPA Gatekeeper

### 이론: 정책 엔진 — OPA Gatekeeper와 Kyverno

모든 정책에 Go로 웹훅을 작성하는 것은 지루해집니다. **정책 엔진**은 선언적 정책을 쿠버네티스 리소스로 읽는 사전 빌드된 validating(때로는 mutating) 웹훅입니다. 두 가지 주류 선택:

**OPA Gatekeeper.** Open Policy Agent 런타임 위에 빌드 — 정책은 선언적 논리 언어인 **Rego**로 작성됩니다. 두 CRD:

- `ConstraintTemplate` — Rego의 매개변수화된 정책 정의. 함수 정의 같은 것.
- `Constraint`(템플릿에서 생성된 커스텀 kind) — 매개변수와 함께한 템플릿 인스턴스. 함수 호출 같은 것.

예 — `RequiredLabels` 템플릿 + "모든 네임스페이스는 `cost-center`와 `team` 레이블이 있어야 한다"는 constraint. Gatekeeper가 이를 웹훅 결정으로 컴파일합니다. 강력한 audit 기능(어드미션뿐 아니라 기존 객체에 대한 지속적 재평가), 그리고 동일한 Rego 정책이 쿠버네티스 외부(Envoy authz, Terraform, 커스텀 앱)에서 재사용 가능.

**Kyverno.** 네이티브 쿠버네티스 — 정책은 YAML로 작성된 CRD(`ClusterPolicy`, `Policy`), 배울 DSL 없음. 세 규칙 유형 — validate(allow/deny), mutate(기본값 설정, 레이블 추가), generate(템플릿에서 자식 리소스 생성).

```yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: require-labels
spec:
  validationFailureAction: enforce
  rules:
    - name: check-team-label
      match:
        any:
          - resources:
              kinds: [Namespace]
      validate:
        message: "Namespace must have a 'team' label"
        pattern:
          metadata:
            labels:
              team: "?*"
```

Kyverno는 접근성에서 이깁니다 — 보안 팀이 Rego를 배우지 않고 정책을 작성할 수 있습니다. OPA는 강력함에서 이깁니다 — Rego는 Kyverno의 선언적 구문에서 어색한 정책을 표현할 수 있습니다.

두 엔진 모두 §B의 동일한 Mutating/ValidatingWebhookConfiguration 머신너리에 플러그인합니다 — 웹훅과 정책 엔진 사이를 선택하는 것이 아니라 웹훅 *으로서* 무엇이 실행되는지를 선택합니다.

### 7.1 Gatekeeper란?

OPA(Open Policy Agent) Gatekeeper는 OPA의 전용 정책 언어인 Rego를 사용하여 정책을 정의할 수 있는 검증 어드미션 웹훅입니다. CRD를 통해 정책을 표현하고 시행하는 Kubernetes 네이티브 방식을 제공합니다.

### 7.2 설치

```bash
# Install Gatekeeper using Helm
helm repo add gatekeeper https://open-policy-agent.github.io/gatekeeper/charts
helm install gatekeeper gatekeeper/gatekeeper \
  --namespace gatekeeper-system \
  --create-namespace \
  --set replicas=3 \
  --set audit.replicas=1 \
  --set audit.logLevel=INFO

# Verify installation
kubectl get pods -n gatekeeper-system
kubectl get crd | grep gatekeeper
```

### 7.3 제약 조건 템플릿

ConstraintTemplate은 Rego 정책 로직과 받아들이는 매개변수를 정의합니다:

```yaml
apiVersion: templates.gatekeeper.sh/v1
kind: ConstraintTemplate
metadata:
  name: k8srequiredlabels
spec:
  crd:
    spec:
      names:
        kind: K8sRequiredLabels
      validation:
        openAPIV3Schema:
          type: object
          properties:
            labels:
              type: array
              description: "List of required label keys"
              items:
                type: string
  targets:
  - target: admission.k8s.gatekeeper.sh
    rego: |
      package k8srequiredlabels

      violation[{"msg": msg}] {
        provided := {label | input.review.object.metadata.labels[label]}
        required := {label | label := input.parameters.labels[_]}
        missing := required - provided
        count(missing) > 0
        msg := sprintf("Missing required labels: %v", [missing])
      }
```

### 7.4 제약 조건

Constraint는 특정 매개변수로 ConstraintTemplate을 인스턴스화합니다:

```yaml
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sRequiredLabels
metadata:
  name: require-team-label
spec:
  enforcementAction: deny  # deny, dryrun, or warn
  match:
    kinds:
    - apiGroups: [""]
      kinds: ["Namespace"]
    - apiGroups: ["apps"]
      kinds: ["Deployment", "StatefulSet", "DaemonSet"]
    excludedNamespaces:
    - kube-system
    - gatekeeper-system
  parameters:
    labels:
    - "team"
    - "cost-center"
```

### 7.5 고급 Rego 정책

**특권 컨테이너(privileged container) 거부:**

```yaml
apiVersion: templates.gatekeeper.sh/v1
kind: ConstraintTemplate
metadata:
  name: k8sdenyprivileged
spec:
  crd:
    spec:
      names:
        kind: K8sDenyPrivileged
  targets:
  - target: admission.k8s.gatekeeper.sh
    rego: |
      package k8sdenyprivileged

      violation[{"msg": msg}] {
        container := input.review.object.spec.containers[_]
        container.securityContext.privileged == true
        msg := sprintf("Privileged container is not allowed: %v", [container.name])
      }

      violation[{"msg": msg}] {
        container := input.review.object.spec.initContainers[_]
        container.securityContext.privileged == true
        msg := sprintf("Privileged init container is not allowed: %v", [container.name])
      }
```

**허용된 레지스트리 제한:**

```yaml
apiVersion: templates.gatekeeper.sh/v1
kind: ConstraintTemplate
metadata:
  name: k8sallowedregistries
spec:
  crd:
    spec:
      names:
        kind: K8sAllowedRegistries
      validation:
        openAPIV3Schema:
          type: object
          properties:
            registries:
              type: array
              items:
                type: string
  targets:
  - target: admission.k8s.gatekeeper.sh
    rego: |
      package k8sallowedregistries

      violation[{"msg": msg}] {
        container := input.review.object.spec.containers[_]
        not registry_allowed(container.image)
        msg := sprintf("Container %v uses image %v from a disallowed registry. Allowed: %v",
          [container.name, container.image, input.parameters.registries])
      }

      violation[{"msg": msg}] {
        container := input.review.object.spec.initContainers[_]
        not registry_allowed(container.image)
        msg := sprintf("Init container %v uses image %v from a disallowed registry. Allowed: %v",
          [container.name, container.image, input.parameters.registries])
      }

      registry_allowed(image) {
        registry := input.parameters.registries[_]
        startswith(image, registry)
      }
---
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sAllowedRegistries
metadata:
  name: allowed-registries
spec:
  enforcementAction: deny
  match:
    kinds:
    - apiGroups: [""]
      kinds: ["Pod"]
    - apiGroups: ["apps"]
      kinds: ["Deployment", "StatefulSet", "DaemonSet"]
  parameters:
    registries:
    - "gcr.io/my-project/"
    - "docker.io/library/"
    - "ghcr.io/my-org/"
```

### 7.6 Gatekeeper 감사

Gatekeeper는 정기적으로 감사(audit)를 실행하여 제약 조건을 위반하는 기존 리소스를 찾습니다:

```bash
# Check constraint violations
kubectl get k8srequiredlabels require-team-label -o yaml

# The status section shows violations
# status:
#   totalViolations: 5
#   violations:
#   - enforcementAction: deny
#     kind: Deployment
#     name: frontend
#     namespace: default
#     message: 'Missing required labels: {"team"}'
```

---

## 8. Kyverno

### 8.1 Kyverno vs Gatekeeper

| 기능 | Gatekeeper | Kyverno |
|---|---|---|
| 정책 언어 | Rego (학습 곡선 있음) | YAML (Kubernetes 네이티브) |
| 변이 지원 | 제한적 (assign/modify) | 완전한 JSON Patch 및 strategic merge |
| 생성(Generation) | 아니오 | 예 (정책에서 리소스 생성) |
| 이미지 검증 | 외부 데이터를 통해 | 내장 (cosign, Notary) |
| 정책 리포트 | 제약 조건 status를 통해 | 전용 PolicyReport CRD |
| CLI 테스팅 | `opa test` + `gator` | `kyverno test` |

### 8.2 설치

```bash
# Install Kyverno using Helm
helm repo add kyverno https://kyverno.github.io/kyverno/
helm install kyverno kyverno/kyverno \
  --namespace kyverno \
  --create-namespace \
  --set replicaCount=3

# Verify
kubectl get pods -n kyverno
```

### 8.3 Kyverno 검증 정책

```yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: require-resource-limits
  annotations:
    policies.kyverno.io/title: Require Resource Limits
    policies.kyverno.io/category: Best Practices
    policies.kyverno.io/severity: medium
spec:
  validationFailureAction: Enforce  # Enforce or Audit
  background: true  # Scan existing resources
  rules:
  - name: check-container-limits
    match:
      any:
      - resources:
          kinds:
          - Pod
    exclude:
      any:
      - resources:
          namespaces:
          - kube-system
    validate:
      message: "CPU and memory limits are required for container {{request.object.spec.containers[*].name}}"
      pattern:
        spec:
          containers:
          - resources:
              limits:
                memory: "?*"
                cpu: "?*"
```

### 8.4 Kyverno 변이 정책

```yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: add-default-labels
spec:
  rules:
  - name: add-labels
    match:
      any:
      - resources:
          kinds:
          - Deployment
          - StatefulSet
    mutate:
      patchStrategicMerge:
        metadata:
          labels:
            +(managed-by): "platform-team"
            +(environment): "{{request.namespace}}"
        spec:
          template:
            metadata:
              labels:
                +(managed-by): "platform-team"
```

### 8.5 Kyverno 생성 정책

Kyverno는 트리거 이벤트가 발생할 때 자동으로 리소스를 생성할 수 있습니다:

```yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: generate-network-policy
spec:
  rules:
  - name: default-deny-ingress
    match:
      any:
      - resources:
          kinds:
          - Namespace
    exclude:
      any:
      - resources:
          names:
          - kube-system
          - kube-public
    generate:
      synchronize: true  # Keep in sync if policy changes
      apiVersion: networking.k8s.io/v1
      kind: NetworkPolicy
      name: default-deny-ingress
      namespace: "{{request.object.metadata.name}}"
      data:
        spec:
          podSelector: {}
          policyTypes:
          - Ingress
```

---

## 9. 어드미션 정책 테스팅

### 9.1 gator로 Gatekeeper 정책 테스팅

```bash
# Install gator CLI
go install github.com/open-policy-agent/gatekeeper/v3/cmd/gator@latest

# Create a test suite
cat > tests/require-labels-test.yaml <<EOF
apiVersion: test.gatekeeper.sh/v1alpha1
kind: Suite
metadata:
  name: require-labels-test
tests:
- name: deployment-without-labels
  template: template.yaml
  constraint: constraint.yaml
  cases:
  - name: should-reject-missing-labels
    object: testdata/deployment-no-labels.yaml
    assertions:
    - violations: 1
      message: "Missing required labels"
  - name: should-allow-with-labels
    object: testdata/deployment-with-labels.yaml
    assertions:
    - violations: 0
EOF

# Run the tests
gator verify tests/
```

### 9.2 CLI로 Kyverno 정책 테스팅

```bash
# Install Kyverno CLI
brew install kyverno  # macOS
# or
kubectl krew install kyverno

# Test a policy against a resource
kyverno apply policy.yaml --resource deployment.yaml

# Run a test suite
cat > kyverno-test.yaml <<EOF
name: require-resource-limits-test
policies:
- require-resource-limits.yaml
resources:
- testdata/pod-with-limits.yaml
- testdata/pod-without-limits.yaml
results:
- policy: require-resource-limits
  rule: check-container-limits
  resource: pod-with-limits
  kind: Pod
  result: pass
- policy: require-resource-limits
  rule: check-container-limits
  resource: pod-without-limits
  kind: Pod
  result: fail
EOF

kyverno test .
```

### 9.3 웹훅 통합 테스팅

```go
// Test webhook with a mock HTTP server
func TestValidateHandler(t *testing.T) {
    pod := corev1.Pod{
        ObjectMeta: metav1.ObjectMeta{Name: "test-pod"},
        Spec: corev1.PodSpec{
            Containers: []corev1.Container{
                {Name: "app", Image: "nginx:latest"},
            },
        },
    }
    podBytes, _ := json.Marshal(pod)

    review := admissionv1.AdmissionReview{
        TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"},
        Request: &admissionv1.AdmissionRequest{
            UID: "test-uid",
            Object: runtime.RawExtension{
                Raw: podBytes,
            },
        },
    }
    reviewBytes, _ := json.Marshal(review)

    req := httptest.NewRequest("POST", "/validate", bytes.NewReader(reviewBytes))
    req.Header.Set("Content-Type", "application/json")
    rec := httptest.NewRecorder()

    handleValidate(rec, req)

    var response admissionv1.AdmissionReview
    json.Unmarshal(rec.Body.Bytes(), &response)

    if response.Response.Allowed {
        t.Error("expected request to be denied for :latest tag")
    }
}
```

### 9.4 Dry-Run 테스팅

```bash
# Test admission with dry-run (does not persist, but still runs webhooks)
kubectl apply --dry-run=server -f pod.yaml

# Test Gatekeeper constraint in dryrun mode first
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sRequiredLabels
metadata:
  name: require-team-label
spec:
  enforcementAction: dryrun  # Will not block, only audit
```

---

## 10. 어드미션 컨트롤러 성능

### 이론: 운영 제약 — 지연, Fail-Policy, Side Effect

웹훅은 API 핫 패스에 삽니다. 제약은 사소하지 않습니다:

- **지연(Latency)** — 모든 API 요청이 모든 매치된 웹훅을 기다립니다. API 서버는 기본 10초 timeout(아래로 구성 가능)을 가집니다. 느린 웹훅은 모든 kubectl apply를 느리게 만듭니다. **권장 — 웹훅은 100ms p99 내에 응답해야 합니다.**
- **Fail policy**는 `Fail`(기본) 또는 `Ignore`. `Fail`이면 웹훅에 도달할 수 없을 때 요청이 거부됩니다 — 엄격하지만 웹훅 장애가 배포를 깨뜨립니다. `Ignore`이면 오류 시 웹훅을 우회 — 우아하지만 장애 동안 정책 위반을 통과시킵니다. 대부분의 보안 웹훅은 `Fail` + 고가용성(여러 레플리카, PDB, 테스트된 롤아웃)을 사용해야 합니다.
- **멱등성(Idempotence)** — 웹훅은 재시도될 수 있습니다 — 특히 `reinvocationPolicy: IfNeeded`인 mutating. `metadata.labels.foo = bar`를 설정하는 웹훅은 멱등입니다. 리스트에 *추가*하는 웹훅("이 사이드카를 컨테이너 리스트에 추가")은 먼저 이미 있는지 검사해야 합니다 — 그렇지 않으면 재시도 시 중복 사이드카.
- **Side effects** — 웹훅은 외부 side effect를 가져서는 안 됩니다(웹훅에서 Slack에 포스팅 금지). API 서버는 오류 시 재시도하고 한 논리적 요청에 대해 웹훅을 여러 번 호출할 수 있습니다. API 서버가 자유롭게 호출할 수 있음을 알도록 `sideEffects: None`(또는 `NoneOnDryRun`)을 사용하세요.
- **Scope** — 웹훅이 관심 있는 것에만 호출되도록 `rules`를 정확히 구성하세요. `pods`를 watch하면서 모든 CRD apply에서 실행되는 웹훅은 그저 지연만 추가합니다.

이 제약들이 프로덕션 웹훅이 보통 프레임워크(Go용 kubebuilder, WebAssembly용 kubewarden)로 작성되어 TLS, 요청 파싱, AdmissionReview 스키마를 처리하고 정책 로직만 남기는 이유입니다.

### 10.1 성능 고려사항

어드미션 웹훅은 규칙에 매칭되는 모든 API 요청에 지연 시간을 추가합니다. 성능이 좋지 않으면 전체 클러스터를 느리게 만들 수 있습니다.

| 요소 | 영향 | 완화 방법 |
|---|---|---|
| 웹훅 지연 시간 | 매칭된 모든 요청에 추가 | 웹훅 로직을 단순하게, 데이터 캐싱 |
| 네트워크 홉 | 다른 노드의 웹훅은 RTT 추가 | 같은 위치에 배치 또는 클러스터 내 웹훅 사용 |
| TLS 핸드셰이크 | 연결당 오버헤드 | HTTP/2 활성화, 연결 풀링 |
| 실패 모드 | `Fail` 정책은 모든 매칭 요청 차단 | 중요하지 않은 웹훅에 `Ignore` 사용 |
| 매칭 범위 | 넓은 규칙은 더 많은 요청 처리 | `rules` 좁히기, namespace/object 셀렉터 사용 |

### 10.2 성능 최적화

```yaml
# Narrow the scope as much as possible
webhooks:
- name: validate.example.com
  rules:
  - operations: ["CREATE"]      # Only CREATE, not UPDATE
    apiGroups: ["apps"]          # Only apps group, not "*"
    apiVersions: ["v1"]          # Specific version
    resources: ["deployments"]   # Specific resource, not "*"
    scope: "Namespaced"          # Skip cluster-scoped resources
  namespaceSelector:
    matchLabels:
      policy-enforcement: enabled   # Only labeled namespaces
  timeoutSeconds: 3                 # Fail fast
  failurePolicy: Ignore            # Do not block on webhook failure
```

### 10.3 웹훅 성능 모니터링

```bash
# Check API server metrics for webhook latency
kubectl get --raw /metrics | grep apiserver_admission_webhook_admission_duration_seconds

# Prometheus query for webhook latency (p99)
# histogram_quantile(0.99,
#   rate(apiserver_admission_webhook_admission_duration_seconds_bucket{
#     name="validate.image-policy.example.com"
#   }[5m])
# )

# Check webhook rejection rate
# sum(rate(apiserver_admission_webhook_rejection_count{
#   name="validate.image-policy.example.com"
# }[5m]))
```

### 10.4 고가용성

```yaml
# Run multiple webhook replicas with pod anti-affinity
apiVersion: apps/v1
kind: Deployment
metadata:
  name: webhook-server
spec:
  replicas: 3
  template:
    spec:
      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: kubernetes.io/hostname
        whenUnsatisfiable: DoNotSchedule
        labelSelector:
          matchLabels:
            app: webhook-server
      containers:
      - name: webhook
        image: example.com/webhook:v1
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 256Mi
```

---

## 연습문제

### 연습문제 1: 검증 웹훅 구축

다음 조건에서 Pod 생성을 거부하는 Go 검증 웹훅 서버를 작성하세요: (a) pod에 `team` 레이블이 없는 경우, (b) 어떤 컨테이너든 root로 실행되는 경우 (securityContext.runAsNonRoot가 false이거나 미설정), (c) 어떤 컨테이너든 4 CPU 코어 이상을 요청하는 경우. ValidatingWebhookConfiguration YAML을 포함하세요.

<details>
<summary>정답 보기</summary>

```go
func validatePodSecurity(pod *corev1.Pod) (bool, string) {
    // Check team label
    if _, ok := pod.Labels["team"]; !ok {
        return false, "pod must have a 'team' label"
    }

    for _, c := range pod.Spec.Containers {
        // Check runAsNonRoot
        if c.SecurityContext == nil || c.SecurityContext.RunAsNonRoot == nil || !*c.SecurityContext.RunAsNonRoot {
            return false, fmt.Sprintf("container %q must set securityContext.runAsNonRoot=true", c.Name)
        }

        // Check CPU limit
        if cpuLimit, ok := c.Resources.Limits[corev1.ResourceCPU]; ok {
            if cpuLimit.Value() > 4 {
                return false, fmt.Sprintf("container %q requests %v CPU, max allowed is 4", c.Name, cpuLimit.String())
            }
        }
    }
    return true, ""
}
```

```yaml
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
metadata:
  name: pod-security-validator
webhooks:
- name: validate.pod-security.example.com
  admissionReviewVersions: ["v1"]
  sideEffects: None
  timeoutSeconds: 5
  failurePolicy: Fail
  clientConfig:
    service:
      name: pod-security-webhook
      namespace: webhook-system
      path: /validate
      port: 443
  rules:
  - operations: ["CREATE"]
    apiGroups: [""]
    apiVersions: ["v1"]
    resources: ["pods"]
  namespaceSelector:
    matchExpressions:
    - key: kubernetes.io/metadata.name
      operator: NotIn
      values: ["kube-system", "webhook-system"]
```

</details>

### 연습문제 2: 변이 웹훅 작성

다음을 수행하는 변이 웹훅을 작성하세요: (a) 모든 Pod에 `prometheus.io/scrape: "true"` 어노테이션 추가, (b) 명시적으로 설정되지 않은 경우 `automountServiceAccountToken: false` 설정, (c) `monitoring` 네임스페이스의 Pod에 `dedicated=monitoring:NoSchedule` 키에 대한 `toleration` 추가. JSON Patch 배열을 반환하세요.

<details>
<summary>정답 보기</summary>

```go
func buildMutationPatches(pod *corev1.Pod, namespace string) []map[string]interface{} {
    patches := []map[string]interface{}{}

    // (a) Add Prometheus scrape annotation
    if pod.Annotations == nil {
        patches = append(patches, map[string]interface{}{
            "op":    "add",
            "path":  "/metadata/annotations",
            "value": map[string]string{"prometheus.io/scrape": "true"},
        })
    } else if _, ok := pod.Annotations["prometheus.io/scrape"]; !ok {
        patches = append(patches, map[string]interface{}{
            "op":    "add",
            "path":  "/metadata/annotations/prometheus.io~1scrape",
            "value": "true",
        })
    }

    // (b) Set automountServiceAccountToken to false if not set
    if pod.Spec.AutomountServiceAccountToken == nil {
        falseVal := false
        _ = falseVal
        patches = append(patches, map[string]interface{}{
            "op":    "add",
            "path":  "/spec/automountServiceAccountToken",
            "value": false,
        })
    }

    // (c) Add monitoring toleration for monitoring namespace
    if namespace == "monitoring" {
        toleration := map[string]interface{}{
            "key":      "dedicated",
            "operator": "Equal",
            "value":    "monitoring",
            "effect":   "NoSchedule",
        }
        if len(pod.Spec.Tolerations) == 0 {
            patches = append(patches, map[string]interface{}{
                "op":    "add",
                "path":  "/spec/tolerations",
                "value": []map[string]interface{}{toleration},
            })
        } else {
            patches = append(patches, map[string]interface{}{
                "op":    "add",
                "path":  "/spec/tolerations/-",
                "value": toleration,
            })
        }
    }

    return patches
}
```

어노테이션이나 toleration이 없는 `monitoring` 네임스페이스의 Pod에 대한 결과 JSON Patch:

```json
[
  {"op": "add", "path": "/metadata/annotations", "value": {"prometheus.io/scrape": "true"}},
  {"op": "add", "path": "/spec/automountServiceAccountToken", "value": false},
  {"op": "add", "path": "/spec/tolerations", "value": [{"key": "dedicated", "operator": "Equal", "value": "monitoring", "effect": "NoSchedule"}]}
]
```

</details>

### 연습문제 3: OPA Gatekeeper 정책

다음을 시행하는 Gatekeeper ConstraintTemplate과 Constraint를 작성하세요: (a) 프로덕션 네임스페이스에서 모든 Deployment는 최소 2개의 레플리카를 가져야 함, (b) 제약 조건은 `env: production` 레이블이 있는 네임스페이스에만 적용, (c) `kube-system` 네임스페이스의 Deployment는 면제.

<details>
<summary>정답 보기</summary>

```yaml
apiVersion: templates.gatekeeper.sh/v1
kind: ConstraintTemplate
metadata:
  name: k8sminreplicas
spec:
  crd:
    spec:
      names:
        kind: K8sMinReplicas
      validation:
        openAPIV3Schema:
          type: object
          properties:
            minReplicas:
              type: integer
              description: "Minimum number of replicas required"
  targets:
  - target: admission.k8s.gatekeeper.sh
    rego: |
      package k8sminreplicas

      violation[{"msg": msg}] {
        input.review.object.kind == "Deployment"
        replicas := object.get(input.review.object.spec, "replicas", 1)
        replicas < input.parameters.minReplicas
        msg := sprintf(
          "Deployment %v has %v replicas, minimum required is %v",
          [input.review.object.metadata.name, replicas, input.parameters.minReplicas]
        )
      }
---
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sMinReplicas
metadata:
  name: production-min-replicas
spec:
  enforcementAction: deny
  match:
    kinds:
    - apiGroups: ["apps"]
      kinds: ["Deployment"]
    namespaceSelector:
      matchLabels:
        env: production
    excludedNamespaces:
    - kube-system
  parameters:
    minReplicas: 2
```

테스트:

```bash
# This should be rejected (1 replica in production namespace)
kubectl -n production-ns create deployment test --image=nginx --replicas=1

# This should be allowed (2+ replicas)
kubectl -n production-ns create deployment test --image=nginx --replicas=3
```

</details>

### 연습문제 4: Kyverno 정책 모음

세 가지 Kyverno ClusterPolicy를 작성하세요: (a) Pod가 `hostNetwork: true`를 사용하는 것을 방지하는 검증 정책, (b) 명시적으로 설정하지 않은 모든 컨테이너에 `readOnlyRootFilesystem: true`를 추가하는 변이 정책, (c) 모든 새 네임스페이스에 ResourceQuota(10 CPU, 20Gi 메모리)를 생성하는 생성 정책.

<details>
<summary>정답 보기</summary>

```yaml
# (a) Deny hostNetwork
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: deny-host-network
spec:
  validationFailureAction: Enforce
  background: true
  rules:
  - name: deny-host-network
    match:
      any:
      - resources:
          kinds:
          - Pod
    exclude:
      any:
      - resources:
          namespaces:
          - kube-system
    validate:
      message: "Using hostNetwork is not allowed. Pod {{request.object.metadata.name}} sets hostNetwork: true"
      pattern:
        spec:
          =(hostNetwork): false
---
# (b) Mutate readOnlyRootFilesystem
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: set-readonly-root-fs
spec:
  rules:
  - name: set-readonly-root
    match:
      any:
      - resources:
          kinds:
          - Pod
    mutate:
      foreach:
      - list: "request.object.spec.containers"
        patchStrategicMerge:
          spec:
            containers:
            - name: "{{element.name}}"
              securityContext:
                +(readOnlyRootFilesystem): true
---
# (c) Generate ResourceQuota
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: generate-resource-quota
spec:
  rules:
  - name: create-default-quota
    match:
      any:
      - resources:
          kinds:
          - Namespace
    exclude:
      any:
      - resources:
          names:
          - kube-system
          - kube-public
          - kube-node-lease
          - kyverno
    generate:
      synchronize: true
      apiVersion: v1
      kind: ResourceQuota
      name: default-quota
      namespace: "{{request.object.metadata.name}}"
      data:
        spec:
          hard:
            requests.cpu: "10"
            requests.memory: 20Gi
            limits.cpu: "20"
            limits.memory: 40Gi
```

</details>

### 연습문제 5: 웹훅 실패 모드

프로덕션 클러스터에 간헐적 타임아웃이 발생하는 검증 웹훅이 있습니다. 다음을 설명하세요: (a) `failurePolicy: Fail`과 `failurePolicy: Ignore`의 차이점과 각각의 사용 시기, (b) 중요 네임스페이스(`kube-system`, `monitoring`)가 절대 차단되지 않도록 웹훅을 구성하는 방법, (c) Prometheus 메트릭을 사용하여 웹훅 지연 시간을 모니터링하는 방법, (d) 적절한 복원력 설정이 포함된 업데이트된 WebhookConfiguration 작성.

<details>
<summary>정답 보기</summary>

**(a) 실패 정책(Failure Policy):**

- `Fail`: 웹훅에 연결할 수 없거나 타임아웃이 발생하면 API 요청이 거부됩니다. 잠재적으로 안전하지 않은 변경을 허용하는 것보다 작업을 차단하는 것이 나은 보안에 중요한 정책에 사용합니다 (예: 이미지 출처 검증).
- `Ignore`: 웹훅에 연결할 수 없거나 타임아웃이 발생하면 API 요청이 웹훅 유효성 검증 없이 진행됩니다. 시행보다 가용성이 더 중요한 중요하지 않은 정책에 사용합니다 (예: 레이블 권장, 비용 추적).

**(b) 네임스페이스 제외**는 `namespaceSelector`를 사용하여 구성합니다:

**(c) 모니터링할 Prometheus 메트릭:**

```promql
# Webhook call latency (p99)
histogram_quantile(0.99,
  rate(apiserver_admission_webhook_admission_duration_seconds_bucket{
    name="validate.example.com",
    type="validating"
  }[5m])
)

# Webhook rejection rate
sum(rate(apiserver_admission_webhook_rejection_count{name="validate.example.com"}[5m]))

# Webhook failure/timeout rate
sum(rate(apiserver_admission_webhook_fail_open_count{name="validate.example.com"}[5m]))
```

**(d) 복원력 있는 구성:**

```yaml
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
metadata:
  name: resilient-validator
webhooks:
- name: validate.example.com
  admissionReviewVersions: ["v1", "v1beta1"]
  sideEffects: None
  timeoutSeconds: 3
  failurePolicy: Ignore
  clientConfig:
    service:
      name: validator
      namespace: webhook-system
      path: /validate
      port: 443
  rules:
  - operations: ["CREATE", "UPDATE"]
    apiGroups: ["apps"]
    apiVersions: ["v1"]
    resources: ["deployments"]
    scope: "Namespaced"
  namespaceSelector:
    matchExpressions:
    - key: kubernetes.io/metadata.name
      operator: NotIn
      values:
      - kube-system
      - kube-public
      - kube-node-lease
      - monitoring
      - webhook-system
    - key: webhook-validation
      operator: In
      values: ["enabled"]
  objectSelector:
    matchExpressions:
    - key: skip-webhook
      operator: DoesNotExist
```

</details>

---

**이전**: [오퍼레이터](./11_Operators.md) | **다음**: [오토스케일링](./13_Autoscaling.md)
