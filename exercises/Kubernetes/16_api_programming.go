// Exercise: Lesson 16 — Kubernetes API Programming
// Complete the TODO items below.

package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"

	// For Exercise 4: Informers
	"k8s.io/client-go/informers"
	"k8s.io/client-go/tools/cache"
)

// === Exercise 1: Initialize the Kubernetes Client ===
// Set up a client-go clientset to interact with the API server.
// Hint: Use kubeconfig from ~/.kube/config or in-cluster config.

func createClient() (*kubernetes.Clientset, error) {
	// TODO: Build the kubeconfig path from home directory
	// Hint: filepath.Join(homedir.HomeDir(), ".kube", "config")
	var kubeconfig string

	// TODO: Build the rest.Config from the kubeconfig file
	// Hint: clientcmd.BuildConfigFromFlags("", kubeconfig)

	// TODO: Create and return the kubernetes.Clientset
	// Hint: kubernetes.NewForConfig(config)

	_ = kubeconfig // remove after implementing
	return nil, fmt.Errorf("TODO: implement createClient")
}

// === Exercise 2: CRUD Operations on Deployments ===
// Create, read, update, and delete a Deployment programmatically.
// Hint: Use clientset.AppsV1().Deployments(namespace) for Deployment operations.

func crudDeployment(clientset *kubernetes.Clientset) error {
	ctx := context.TODO()
	namespace := "default"
	deploymentsClient := clientset.AppsV1().Deployments(namespace)

	// Step 1: Create a Deployment
	replicas := int32(3)
	deployment := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name: "programmatic-deploy",
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"app": "demo"},
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"app": "demo"},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  "web",
							Image: "nginx:1.25-alpine",
							Ports: []corev1.ContainerPort{
								// TODO: Add a ContainerPort for port 80
							},
						},
					},
				},
			},
		},
	}

	// TODO: Create the Deployment using deploymentsClient.Create()
	// Hint: deploymentsClient.Create(ctx, deployment, metav1.CreateOptions{})

	// Step 2: Read the Deployment
	// TODO: Fetch the deployment using deploymentsClient.Get()
	// TODO: Print the number of replicas and image

	// Step 3: Update — scale to 5 replicas
	// TODO: Set the fetched deployment's replicas to 5
	// TODO: Call deploymentsClient.Update(ctx, deploy, metav1.UpdateOptions{})

	// Step 4: List all Deployments
	// TODO: Use deploymentsClient.List(ctx, metav1.ListOptions{})
	// TODO: Print each Deployment name and replicas

	// Step 5: Delete the Deployment
	// TODO: Use deploymentsClient.Delete(ctx, "programmatic-deploy", metav1.DeleteOptions{})

	_ = ctx       // remove after implementing
	_ = namespace // remove after implementing
	return nil
}

// === Exercise 3: Watch for Pod Events ===
// Set up a watch to monitor Pod lifecycle events in real time.
// Hint: Watch returns a channel of Events (ADDED, MODIFIED, DELETED).

func watchPods(clientset *kubernetes.Clientset) error {
	ctx := context.TODO()

	// TODO: Create a watcher using clientset.CoreV1().Pods("").Watch()
	// Hint: Use metav1.ListOptions{LabelSelector: "app=demo"}
	// to watch only Pods with the "app=demo" label

	// TODO: Iterate over the watch channel (watcher.ResultChan())
	// For each event:
	//   - Type cast event.Object to *corev1.Pod
	//   - Print the event type (ADDED/MODIFIED/DELETED)
	//   - Print the Pod name, phase, and conditions
	// Hint:
	//   for event := range watcher.ResultChan() {
	//       pod, ok := event.Object.(*corev1.Pod)
	//       if !ok { continue }
	//       fmt.Printf("Event: %s Pod: %s Phase: %s\n",
	//           event.Type, pod.Name, pod.Status.Phase)
	//   }

	// TODO: Add a timeout using context.WithTimeout(ctx, 5*time.Minute)
	// to avoid watching forever

	_ = ctx // remove after implementing
	return nil
}

// === Exercise 4: Shared Informers (Cached Watch) ===
// Use informers for efficient, cached access to cluster state.
// Hint: Informers maintain a local cache and only sync deltas.

func useInformers(clientset *kubernetes.Clientset) {
	// TODO: Create a shared informer factory with 30-second resync
	// Hint: informers.NewSharedInformerFactory(clientset, 30*time.Second)

	// TODO: Get a Pod informer from the factory
	// Hint: factory.Core().V1().Pods().Informer()

	// TODO: Add event handlers for Add, Update, Delete
	// Hint:
	//   podInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
	//       AddFunc: func(obj interface{}) {
	//           pod := obj.(*corev1.Pod)
	//           fmt.Printf("Pod added: %s/%s\n", pod.Namespace, pod.Name)
	//       },
	//       UpdateFunc: func(oldObj, newObj interface{}) {
	//           oldPod := oldObj.(*corev1.Pod)
	//           newPod := newObj.(*corev1.Pod)
	//           fmt.Printf("Pod updated: %s (phase: %s -> %s)\n",
	//               newPod.Name, oldPod.Status.Phase, newPod.Status.Phase)
	//       },
	//       DeleteFunc: func(obj interface{}) {
	//           pod := obj.(*corev1.Pod)
	//           fmt.Printf("Pod deleted: %s/%s\n", pod.Namespace, pod.Name)
	//       },
	//   })

	// TODO: Start the factory and wait for cache sync
	// Hint:
	//   stopCh := make(chan struct{})
	//   defer close(stopCh)
	//   factory.Start(stopCh)
	//   factory.WaitForCacheSync(stopCh)

	// TODO: Use the lister to query cached data (no API call)
	// Hint: factory.Core().V1().Pods().Lister().Pods("default").List(labels.Everything())
}

// === Exercise 5: Dynamic Client for Custom Resources ===
// Use the dynamic client to work with CRDs without generated types.
// Hint: Dynamic client uses unstructured.Unstructured for any resource.

func dynamicClientExample(clientset *kubernetes.Clientset) error {
	// TODO: Create a dynamic client
	// Hint:
	//   import "k8s.io/client-go/dynamic"
	//   config, _ := clientcmd.BuildConfigFromFlags("", kubeconfig)
	//   dynClient, _ := dynamic.NewForConfig(config)

	// TODO: Define the GVR (GroupVersionResource) for your custom resource
	// Hint:
	//   import "k8s.io/apimachinery/pkg/runtime/schema"
	//   gvr := schema.GroupVersionResource{
	//       Group:    "apps.example.com",
	//       Version:  "v1",
	//       Resource: "webapplications",
	//   }

	// TODO: Create an unstructured custom resource
	// Hint:
	//   import "k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	//   obj := &unstructured.Unstructured{
	//       Object: map[string]interface{}{
	//           "apiVersion": "apps.example.com/v1",
	//           "kind":       "WebApplication",
	//           "metadata":   map[string]interface{}{"name": "test-app"},
	//           "spec":       map[string]interface{}{"image": "nginx", "replicas": 3},
	//       },
	//   }
	//   dynClient.Resource(gvr).Namespace("default").Create(ctx, obj, metav1.CreateOptions{})

	// TODO: List all custom resources of this type
	// Hint: dynClient.Resource(gvr).Namespace("default").List(ctx, metav1.ListOptions{})

	return nil
}

func main() {
	// TODO: Initialize the client
	// TODO: Run the exercises sequentially
	// Hint:
	//   clientset, err := createClient()
	//   if err != nil { panic(err) }
	//   crudDeployment(clientset)
	//   watchPods(clientset)
	//   useInformers(clientset)

	fmt.Println("Kubernetes API programming exercises")
}
