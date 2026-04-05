// 11_operators — Simple operator reconciliation loop skeleton
// Build: go build -o operator 11_operators.go
// Requires: k8s.io/client-go, sigs.k8s.io/controller-runtime

package main

import (
	"context"
	"fmt"
	"os"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
)

// WebAppSpec defines the desired state of WebApp
type WebAppSpec struct {
	Image    string `json:"image"`
	Replicas int32  `json:"replicas"`
	Port     int32  `json:"port"`
}

// WebAppStatus defines the observed state
type WebAppStatus struct {
	ReadyReplicas int32  `json:"readyReplicas"`
	Phase         string `json:"phase"`
}

// WebApp is the Schema for the webapps API
type WebApp struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`
	Spec              WebAppSpec   `json:"spec,omitempty"`
	Status            WebAppStatus `json:"status,omitempty"`
}

// WebAppReconciler reconciles a WebApp object
type WebAppReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

// Reconcile handles create/update/delete events for WebApp resources
func (r *WebAppReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// Fetch the WebApp instance
	webapp := &WebApp{}
	if err := r.Get(ctx, req.NamespacedName, webapp); err != nil {
		if errors.IsNotFound(err) {
			logger.Info("WebApp deleted, cleaning up")
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	logger.Info("Reconciling WebApp",
		"name", webapp.Name,
		"replicas", webapp.Spec.Replicas,
	)

	// Ensure the Deployment exists and matches spec
	deploy := &appsv1.Deployment{}
	deployName := fmt.Sprintf("%s-deployment", webapp.Name)
	err := r.Get(ctx, client.ObjectKey{
		Namespace: webapp.Namespace,
		Name:      deployName,
	}, deploy)

	if errors.IsNotFound(err) {
		// Create the Deployment
		deploy = r.buildDeployment(webapp, deployName)
		if err := r.Create(ctx, deploy); err != nil {
			logger.Error(err, "Failed to create Deployment")
			return ctrl.Result{}, err
		}
		logger.Info("Created Deployment", "name", deployName)
	} else if err == nil {
		// Update if spec changed
		if *deploy.Spec.Replicas != webapp.Spec.Replicas {
			deploy.Spec.Replicas = &webapp.Spec.Replicas
			if err := r.Update(ctx, deploy); err != nil {
				return ctrl.Result{}, err
			}
			logger.Info("Updated Deployment replicas", "replicas", webapp.Spec.Replicas)
		}
	} else {
		return ctrl.Result{}, err
	}

	// Update status
	webapp.Status.ReadyReplicas = deploy.Status.ReadyReplicas
	webapp.Status.Phase = "Running"
	if err := r.Status().Update(ctx, webapp); err != nil {
		logger.Error(err, "Failed to update status")
	}

	return ctrl.Result{}, nil
}

// buildDeployment creates a Deployment for the WebApp
func (r *WebAppReconciler) buildDeployment(webapp *WebApp, name string) *appsv1.Deployment {
	labels := map[string]string{"app": webapp.Name, "managed-by": "webapp-operator"}
	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: webapp.Namespace,
			Labels:    labels,
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &webapp.Spec.Replicas,
			Selector: &metav1.LabelSelector{MatchLabels: labels},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: labels},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name:  "app",
						Image: webapp.Spec.Image,
						Ports: []corev1.ContainerPort{{ContainerPort: webapp.Spec.Port}},
					}},
				},
			},
		},
	}
}

// SetupWithManager registers the controller with the manager
func (r *WebAppReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&WebApp{}).
		Owns(&appsv1.Deployment{}).
		Complete(r)
}

func main() {
	log.SetLogger(zap.New())
	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{})
	if err != nil {
		fmt.Fprintf(os.Stderr, "unable to create manager: %v\n", err)
		os.Exit(1)
	}

	if err := (&WebAppReconciler{
		Client: mgr.GetClient(),
		Scheme: mgr.GetScheme(),
	}).SetupWithManager(mgr); err != nil {
		fmt.Fprintf(os.Stderr, "unable to setup controller: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("Starting operator manager...")
	if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
		fmt.Fprintf(os.Stderr, "manager exited with error: %v\n", err)
		os.Exit(1)
	}
}
