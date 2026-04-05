//go:build ignore

// Exercise: Lesson 11 — Kubernetes Operators
// Complete the TODO items below.

package main

import (
	"context"
	"fmt"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// === Exercise 1: Define the Custom Resource Spec ===
// Define the spec and status for a WebApp custom resource.
// Hint: These structs map to the CRD schema fields.

// WebAppSpec defines the desired state of WebApp
type WebAppSpec struct {
	// TODO: Add field "Image" of type string with json tag "image"
	// TODO: Add field "Replicas" of type *int32 with json tag "replicas"
	// TODO: Add field "Port" of type int32 with json tag "port"
}

// WebAppStatus defines the observed state of WebApp
type WebAppStatus struct {
	// TODO: Add field "AvailableReplicas" of type int32 with json tag "availableReplicas"
	// TODO: Add field "Conditions" of type []metav1.Condition with json tag "conditions"
}

// WebApp is the Schema for the webapps API
type WebApp struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   WebAppSpec   `json:"spec,omitempty"`
	Status WebAppStatus `json:"status,omitempty"`
}

// WebAppList contains a list of WebApp
type WebAppList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []WebApp `json:"items"`
}

// === Exercise 2: Implement the Reconciler ===
// The reconciler is the core logic that reacts to changes.
// Hint: Reconcile is called whenever a WebApp CR is created, updated, or deleted.

// WebAppReconciler reconciles a WebApp object
type WebAppReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

func (r *WebAppReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// Step 1: Fetch the WebApp instance
	webapp := &WebApp{}
	// TODO: Use r.Get() to fetch the WebApp CR
	// Hint: r.Get(ctx, req.NamespacedName, webapp)
	// TODO: If the resource is not found (errors.IsNotFound), return without error
	// TODO: If another error occurs, return the error

	logger.Info("Reconciling WebApp", "name", webapp.Name)

	// Step 2: Create or update the Deployment
	// TODO: Call r.reconcileDeployment(ctx, webapp)
	// TODO: If error, return ctrl.Result{}, err

	// Step 3: Create or update the Service
	// TODO: Call r.reconcileService(ctx, webapp)
	// TODO: If error, return ctrl.Result{}, err

	// Step 4: Update status
	// TODO: Call r.updateStatus(ctx, webapp)
	// TODO: If error, return ctrl.Result{}, err

	return ctrl.Result{}, nil
}

// === Exercise 3: Create the Owned Deployment ===
// Build a Deployment that the operator manages.
// Hint: Use ctrl.SetControllerReference so the Deployment is garbage-collected
// when the WebApp CR is deleted.

func (r *WebAppReconciler) reconcileDeployment(ctx context.Context, webapp *WebApp) error {
	deploy := &appsv1.Deployment{}
	deployName := types.NamespacedName{
		Name:      webapp.Name + "-deployment",
		Namespace: webapp.Namespace,
	}

	err := r.Get(ctx, deployName, deploy)
	if err != nil && errors.IsNotFound(err) {
		// Deployment does not exist — create it
		deploy = &appsv1.Deployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      deployName.Name,
				Namespace: deployName.Namespace,
			},
			Spec: appsv1.DeploymentSpec{
				// TODO: Set Replicas to webapp.Spec.Replicas
				Selector: &metav1.LabelSelector{
					MatchLabels: map[string]string{"app": webapp.Name},
				},
				Template: corev1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{
						Labels: map[string]string{"app": webapp.Name},
					},
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{
							{
								// TODO: Set Name to webapp.Name
								// TODO: Set Image to webapp.Spec.Image
								Ports: []corev1.ContainerPort{
									// TODO: Add a ContainerPort with webapp.Spec.Port
								},
							},
						},
					},
				},
			},
		}

		// TODO: Set the owner reference so the Deployment is cleaned up
		//       when the WebApp CR is deleted
		// Hint: ctrl.SetControllerReference(webapp, deploy, r.Scheme)

		// TODO: Create the Deployment using r.Create(ctx, deploy)
		return nil
	} else if err != nil {
		return err
	}

	// Deployment exists — update if needed
	// TODO: Check if deploy.Spec.Replicas differs from webapp.Spec.Replicas
	// TODO: Check if the container image differs from webapp.Spec.Image
	// TODO: If either changed, call r.Update(ctx, deploy)

	return nil
}

// === Exercise 4: Reconcile the Service ===
// Ensure a ClusterIP Service exists for the WebApp.
// Hint: Services are generally created once and rarely updated.

func (r *WebAppReconciler) reconcileService(ctx context.Context, webapp *WebApp) error {
	svc := &corev1.Service{}
	svcName := types.NamespacedName{
		Name:      webapp.Name + "-service",
		Namespace: webapp.Namespace,
	}

	err := r.Get(ctx, svcName, svc)
	if err != nil && errors.IsNotFound(err) {
		// TODO: Create a new Service with:
		//   - Type: ClusterIP
		//   - Selector: {"app": webapp.Name}
		//   - Port: webapp.Spec.Port -> targetPort: webapp.Spec.Port
		// Hint: Build a corev1.Service{} similar to the Deployment above
		// TODO: Set owner reference with ctrl.SetControllerReference
		// TODO: Create with r.Create(ctx, svc)
		return nil
	}

	return err
}

// === Exercise 5: Update Status Subresource ===
// Report the observed state back to the WebApp status.
// Hint: Use r.Status().Update() to write only the status subresource.

func (r *WebAppReconciler) updateStatus(ctx context.Context, webapp *WebApp) error {
	deploy := &appsv1.Deployment{}
	deployName := types.NamespacedName{
		Name:      webapp.Name + "-deployment",
		Namespace: webapp.Namespace,
	}

	// TODO: Fetch the Deployment to read its status
	// Hint: r.Get(ctx, deployName, deploy)

	// TODO: Set webapp.Status.AvailableReplicas = deploy.Status.AvailableReplicas

	// TODO: Update the status using r.Status().Update(ctx, webapp)
	// Hint: Only the /status subresource is written; spec is unchanged

	_ = deployName // remove after implementing
	_ = deploy     // remove after implementing
	return nil
}

// SetupWithManager registers the controller with the manager.
func (r *WebAppReconciler) SetupWithManager(mgr ctrl.Manager) error {
	// TODO: Complete the controller setup
	// Hint:
	//   return ctrl.NewControllerManagedBy(mgr).
	//       For(&WebApp{}).
	//       Owns(&appsv1.Deployment{}).
	//       Owns(&corev1.Service{}).
	//       Complete(r)
	return nil
}
