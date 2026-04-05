// 16_api_programming — client-go informer and controller example
// Build: go build -o controller 16_api_programming.go
// Requires: k8s.io/client-go, k8s.io/apimachinery

package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/workqueue"
)

// PodController watches Pod events via informer and processes via work queue
type PodController struct {
	clientset kubernetes.Interface
	informer  cache.SharedIndexInformer
	queue     workqueue.RateLimitingInterface
}

// NewPodController creates a controller with informer and work queue
func NewPodController(clientset kubernetes.Interface) *PodController {
	factory := informers.NewSharedInformerFactoryWithOptions(
		clientset, 30*time.Second,
		informers.WithNamespace("default"),
	)
	informer := factory.Core().V1().Pods().Informer()
	queue := workqueue.NewRateLimitingQueue(
		workqueue.DefaultControllerRateLimiter(),
	)

	ctrl := &PodController{
		clientset: clientset,
		informer:  informer,
		queue:     queue,
	}

	// Register event handlers
	informer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			key, err := cache.MetaNamespaceKeyFunc(obj)
			if err == nil {
				fmt.Printf("[ADD] Pod: %s\n", key)
				queue.Add(key)
			}
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			key, err := cache.MetaNamespaceKeyFunc(newObj)
			if err == nil {
				fmt.Printf("[UPDATE] Pod: %s\n", key)
				queue.Add(key)
			}
		},
		DeleteFunc: func(obj interface{}) {
			key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(obj)
			if err == nil {
				fmt.Printf("[DELETE] Pod: %s\n", key)
				queue.Add(key)
			}
		},
	})

	return ctrl
}

// Run starts the controller loop
func (c *PodController) Run(stopCh <-chan struct{}) {
	defer c.queue.ShutDown()

	fmt.Println("Starting PodController...")
	go c.informer.Run(stopCh)

	// Wait for cache sync before processing
	if !cache.WaitForCacheSync(stopCh, c.informer.HasSynced) {
		fmt.Println("Error: cache sync failed")
		return
	}
	fmt.Println("Cache synced, starting workers...")

	// Run 2 worker goroutines
	for i := 0; i < 2; i++ {
		go wait.Until(c.worker, time.Second, stopCh)
	}

	<-stopCh
	fmt.Println("Controller stopped")
}

// worker processes items from the queue
func (c *PodController) worker() {
	for c.processNextItem() {
	}
}

func (c *PodController) processNextItem() bool {
	key, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(key)

	err := c.reconcile(key.(string))
	if err != nil {
		fmt.Printf("Error reconciling %s: %v\n", key, err)
		c.queue.AddRateLimited(key)
		return true
	}

	c.queue.Forget(key)
	return true
}

// reconcile handles a single Pod key
func (c *PodController) reconcile(key string) error {
	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}

	pod, err := c.clientset.CoreV1().Pods(namespace).Get(
		context.TODO(), name, metav1.GetOptions{},
	)
	if err != nil {
		return nil // Pod deleted, nothing to do
	}

	fmt.Printf("  Reconcile: %s/%s phase=%s containers=%d\n",
		pod.Namespace, pod.Name, pod.Status.Phase, len(pod.Spec.Containers))

	// Example: ensure pod has a label
	if _, ok := pod.Labels["watched-by"]; !ok {
		podCopy := pod.DeepCopy()
		if podCopy.Labels == nil {
			podCopy.Labels = make(map[string]string)
		}
		podCopy.Labels["watched-by"] = "pod-controller"
		_, err := c.clientset.CoreV1().Pods(namespace).Update(
			context.TODO(), podCopy, metav1.UpdateOptions{},
		)
		if err != nil {
			return fmt.Errorf("failed to update pod labels: %w", err)
		}
		fmt.Printf("  Added label to %s/%s\n", namespace, name)
	}

	return nil
}

func main() {
	// Build kubeconfig from default location
	home, _ := os.UserHomeDir()
	kubeconfig := filepath.Join(home, ".kube", "config")

	config, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error building config: %v\n", err)
		os.Exit(1)
	}

	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating client: %v\n", err)
		os.Exit(1)
	}

	// List pods to verify connectivity
	pods, _ := clientset.CoreV1().Pods("default").List(
		context.TODO(), metav1.ListOptions{},
	)
	fmt.Printf("Found %d pods in default namespace\n", len(pods.Items))

	// Start controller
	ctrl := NewPodController(clientset)
	stopCh := make(chan struct{})
	defer close(stopCh)
	ctrl.Run(stopCh)

	// Suppress unused import warning
	_ = corev1.Pod{}
}
