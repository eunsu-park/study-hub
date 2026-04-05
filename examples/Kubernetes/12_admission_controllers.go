// 12_admission_controllers — Validating webhook handler skeleton
// Build: go build -o webhook-server 12_admission_controllers.go
// Requires: k8s.io/api, k8s.io/apimachinery, sigs.k8s.io/controller-runtime

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"

	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// PodValidator validates Pod creation requests
type PodValidator struct {
	decoder *admission.Decoder
}

// Handle processes admission requests for Pod resources
func (v *PodValidator) Handle(ctx context.Context, req admission.Request) admission.Response {
	pod := &corev1.Pod{}
	if err := v.decoder.Decode(req, pod); err != nil {
		return admission.Errored(http.StatusBadRequest, err)
	}

	// Rule 1: Pods must have resource limits
	for _, c := range pod.Spec.Containers {
		if c.Resources.Limits.Cpu().IsZero() || c.Resources.Limits.Memory().IsZero() {
			return admission.Denied(fmt.Sprintf(
				"container %q must have CPU and memory limits set", c.Name,
			))
		}
	}

	// Rule 2: No latest tag allowed
	for _, c := range pod.Spec.Containers {
		if strings.HasSuffix(c.Image, ":latest") || !strings.Contains(c.Image, ":") {
			return admission.Denied(fmt.Sprintf(
				"container %q image %q must use explicit tag (not :latest)", c.Name, c.Image,
			))
		}
	}

	// Rule 3: Must have required labels
	requiredLabels := []string{"app", "team"}
	for _, label := range requiredLabels {
		if _, ok := pod.Labels[label]; !ok {
			return admission.Denied(fmt.Sprintf(
				"pod must have label %q", label,
			))
		}
	}

	// Rule 4: No privileged containers
	for _, c := range pod.Spec.Containers {
		if c.SecurityContext != nil && c.SecurityContext.Privileged != nil && *c.SecurityContext.Privileged {
			return admission.Denied(fmt.Sprintf(
				"container %q must not run in privileged mode", c.Name,
			))
		}
	}

	return admission.Allowed("pod passes all validation rules")
}

// PodMutator injects default labels and security settings
type PodMutator struct {
	decoder *admission.Decoder
}

// Handle processes admission requests and mutates Pod resources
func (m *PodMutator) Handle(ctx context.Context, req admission.Request) admission.Response {
	pod := &corev1.Pod{}
	if err := m.decoder.Decode(req, pod); err != nil {
		return admission.Errored(http.StatusBadRequest, err)
	}

	// Mutation 1: Add default labels
	if pod.Labels == nil {
		pod.Labels = make(map[string]string)
	}
	if _, ok := pod.Labels["managed-by"]; !ok {
		pod.Labels["managed-by"] = "admission-webhook"
	}

	// Mutation 2: Set runAsNonRoot if not specified
	if pod.Spec.SecurityContext == nil {
		pod.Spec.SecurityContext = &corev1.PodSecurityContext{}
	}
	if pod.Spec.SecurityContext.RunAsNonRoot == nil {
		nonRoot := true
		pod.Spec.SecurityContext.RunAsNonRoot = &nonRoot
	}

	// Mutation 3: Drop ALL capabilities on each container
	for i := range pod.Spec.Containers {
		if pod.Spec.Containers[i].SecurityContext == nil {
			pod.Spec.Containers[i].SecurityContext = &corev1.SecurityContext{}
		}
		if pod.Spec.Containers[i].SecurityContext.Capabilities == nil {
			pod.Spec.Containers[i].SecurityContext.Capabilities = &corev1.Capabilities{}
		}
		pod.Spec.Containers[i].SecurityContext.Capabilities.Drop = []corev1.Capability{"ALL"}
	}

	// Create JSON patch
	marshaledPod, err := json.Marshal(pod)
	if err != nil {
		return admission.Errored(http.StatusInternalServerError, err)
	}

	return admission.PatchResponseFromRaw(req.Object.Raw, marshaledPod)
}

func main() {
	// In production, use controller-runtime's webhook.Server with TLS certs
	// This skeleton shows the handler logic; see controller-runtime docs for
	// full setup with certificate management.

	fmt.Println("Admission webhook server skeleton")
	fmt.Println("Register handlers with controller-runtime webhook.Server:")
	fmt.Println("  mgr.GetWebhookServer().Register(\"/validate-pods\", &webhook.Admission{Handler: &PodValidator{}})")
	fmt.Println("  mgr.GetWebhookServer().Register(\"/mutate-pods\", &webhook.Admission{Handler: &PodMutator{}})")

	// Prevent unused import errors in this skeleton
	_ = os.Stdout
	_ = http.StatusOK
}
