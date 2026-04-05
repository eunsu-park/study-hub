# 10. Custom Resource Definitions

**Previous**: [Helm and Kustomize](./09_Helm_and_Kustomize.md) | **Next**: [Operators](./11_Operators.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain how Custom Resource Definitions extend the Kubernetes API
2. Create CRDs with structural schemas, validation rules, and default values
3. Implement CRD versioning with storage versions and conversion webhooks
4. Use status and scale subresources with custom printer columns
5. Evaluate when to use CRDs versus aggregated API servers

---

One of Kubernetes' most powerful features is its extensibility. While the core API provides resources like Pods, Services, and Deployments, Custom Resource Definitions (CRDs) let you teach Kubernetes about entirely new resource types -- databases, certificates, network policies, workflow steps, or any domain-specific concept your organization needs. Once registered, custom resources behave exactly like built-in resources: they are persisted in etcd, have RESTful API endpoints, support RBAC, and work with kubectl. This lesson covers CRD design from basic definitions through advanced features like versioning, conversion webhooks, and subresources.

> **The Extension Spectrum:** CRDs are the simplest way to extend the Kubernetes API. They require no Go code to define -- just a YAML manifest. For more complex needs (custom storage backends, authentication, or API aggregation), you can build aggregated API servers. Most use cases are well-served by CRDs combined with a controller (covered in the Operators lesson).

## Table of Contents

- [1. Extending the Kubernetes API](#1-extending-the-kubernetes-api)
  - [1.1 Why Extend Kubernetes?](#11-why-extend-kubernetes)
  - [1.2 Extension Mechanisms](#12-extension-mechanisms)
  - [1.3 How CRDs Work](#13-how-crds-work)
- [2. CRD Specification](#2-crd-specification)
  - [2.1 Basic CRD](#21-basic-crd)
  - [2.2 Creating and Using Custom Resources](#22-creating-and-using-custom-resources)
- [3. Structural Schemas and Validation](#3-structural-schemas-and-validation)
  - [3.1 OpenAPI v3 Schema](#31-openapi-v3-schema)
  - [3.2 Validation Rules (CEL)](#32-validation-rules-cel)
  - [3.3 Default Values](#33-default-values)
  - [3.4 Enum and Pattern Constraints](#34-enum-and-pattern-constraints)
- [4. CRD Versioning](#4-crd-versioning)
  - [4.1 Multiple Versions](#41-multiple-versions)
  - [4.2 Storage Version](#42-storage-version)
  - [4.3 Conversion Webhooks](#43-conversion-webhooks)
- [5. Subresources](#5-subresources)
  - [5.1 Status Subresource](#51-status-subresource)
  - [5.2 Scale Subresource](#52-scale-subresource)
- [6. Printer Columns](#6-printer-columns)
- [7. Categories and Short Names](#7-categories-and-short-names)
- [8. CRD Best Practices](#8-crd-best-practices)
  - [8.1 Design Guidelines](#81-design-guidelines)
  - [8.2 Schema Evolution](#82-schema-evolution)
  - [8.3 Performance Considerations](#83-performance-considerations)
- [9. Aggregated API Servers vs CRDs](#9-aggregated-api-servers-vs-crds)
  - [9.1 When to Use Each](#91-when-to-use-each)
  - [9.2 Aggregated API Server Example](#92-aggregated-api-server-example)
- [Exercises](#exercises)

---

## 1. Extending the Kubernetes API

### 1.1 Why Extend Kubernetes?

Custom resources allow you to represent domain-specific concepts in the Kubernetes API:

| Domain | Custom Resource | What It Represents |
|--------|----------------|-------------------|
| Databases | `PostgresCluster` | A managed PostgreSQL cluster |
| Certificates | `Certificate` | A TLS certificate request (cert-manager) |
| CI/CD | `Pipeline` | A CI/CD pipeline definition (Tekton) |
| Networking | `Gateway` | An L4/L7 load balancer (Gateway API) |
| ML | `TFJob` | A TensorFlow training job (Kubeflow) |
| GitOps | `Application` | An ArgoCD application |

### 1.2 Extension Mechanisms

```
┌──────────────────────────────────────────────────────────────┐
│  Kubernetes API Extension Mechanisms                         │
│                                                              │
│  Simple ─────────────────────────────────────────── Complex  │
│                                                              │
│  ConfigMaps    CRDs         CRDs +          Aggregated       │
│  (data only,   (declarative  Controller     API Server       │
│   no schema)    schema,      (reconciliation (custom storage, │
│                 validation)   loop)          custom auth)     │
│                                                              │
│  Use case:     Use case:     Use case:      Use case:        │
│  App config    API objects   Operators       metrics-server   │
│                               (cert-manager,  (custom backend)│
│                               ArgoCD)                        │
└──────────────────────────────────────────────────────────────┘
```

### 1.3 How CRDs Work

When you create a CRD, the API server dynamically registers new REST endpoints:

```
1. Apply CRD manifest
   ┌──────────┐    POST /apis/apiextensions.k8s.io/v1/customresourcedefinitions
   │  kubectl  │──────────────────────────────────────────────────────────────▶
   └──────────┘

2. API server creates endpoints
   /apis/<group>/<version>/namespaces/<ns>/<plural>
   /apis/<group>/<version>/namespaces/<ns>/<plural>/<name>
   /apis/<group>/<version>/namespaces/<ns>/<plural>/<name>/status
   /apis/<group>/<version>/namespaces/<ns>/<plural>/<name>/scale

3. Custom resources are stored in etcd
   key: /registry/<group>/<plural>/<namespace>/<name>

4. Standard tooling works immediately
   kubectl get <plural>
   kubectl describe <plural> <name>
   kubectl delete <plural> <name>
```

---

## 2. CRD Specification

### 2.1 Basic CRD

```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: databases.example.com       # Must be <plural>.<group>
spec:
  group: example.com                 # API group
  names:
    kind: Database                   # CamelCase singular
    listKind: DatabaseList
    singular: database               # lowercase singular (kubectl)
    plural: databases                # lowercase plural (API path)
    shortNames:                      # Short names for kubectl
      - db
    categories:                      # Grouping for kubectl get all
      - all
      - example
  scope: Namespaced                  # or Cluster
  versions:
    - name: v1alpha1
      served: true                   # Accept requests for this version
      storage: true                  # Store in etcd in this version
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              required:
                - engine
                - version
              properties:
                engine:
                  type: string
                  enum: ["postgres", "mysql", "mongodb"]
                version:
                  type: string
                replicas:
                  type: integer
                  minimum: 1
                  maximum: 10
                  default: 1
                storage:
                  type: object
                  properties:
                    size:
                      type: string
                      pattern: "^[0-9]+(Gi|Ti)$"
                      default: "10Gi"
                    storageClassName:
                      type: string
            status:
              type: object
              properties:
                phase:
                  type: string
                readyReplicas:
                  type: integer
                conditions:
                  type: array
                  items:
                    type: object
                    properties:
                      type:
                        type: string
                      status:
                        type: string
                        enum: ["True", "False", "Unknown"]
                      lastTransitionTime:
                        type: string
                        format: date-time
                      reason:
                        type: string
                      message:
                        type: string
```

```bash
# Apply the CRD
kubectl apply -f database-crd.yaml

# Verify the CRD is registered
kubectl get crd databases.example.com
# NAME                     CREATED AT
# databases.example.com    2024-01-15T10:00:00Z

# Check the new API endpoints
kubectl api-resources | grep database
# databases   db    example.com/v1alpha1   true   Database
```

### 2.2 Creating and Using Custom Resources

```yaml
# my-database.yaml
apiVersion: example.com/v1alpha1
kind: Database
metadata:
  name: orders-db
  namespace: production
spec:
  engine: postgres
  version: "16.2"
  replicas: 3
  storage:
    size: 100Gi
    storageClassName: fast-ssd
```

```bash
# Create a custom resource
kubectl apply -f my-database.yaml

# List databases
kubectl get databases -n production
# or use the short name
kubectl get db -n production
# NAME        AGE
# orders-db   5s

# Describe the resource
kubectl describe db orders-db -n production

# Get as YAML
kubectl get db orders-db -n production -o yaml

# Delete
kubectl delete db orders-db -n production

# Watch for changes
kubectl get db -n production -w
```

---

## 3. Structural Schemas and Validation

### 3.1 OpenAPI v3 Schema

Every CRD must have a structural schema. This schema validates all custom resources at creation and update time.

```yaml
schema:
  openAPIV3Schema:
    type: object
    description: "Database represents a managed database instance"
    properties:
      spec:
        type: object
        description: "Desired state of the database"
        required:
          - engine
          - version
        properties:
          engine:
            type: string
            description: "Database engine type"
            enum: ["postgres", "mysql", "mongodb"]
          version:
            type: string
            description: "Engine version"
          replicas:
            type: integer
            description: "Number of database replicas"
            minimum: 1
            maximum: 10
            default: 1
          storage:
            type: object
            description: "Storage configuration"
            properties:
              size:
                type: string
                description: "Storage size (e.g., 10Gi, 1Ti)"
                pattern: "^[0-9]+(Mi|Gi|Ti)$"
                default: "10Gi"
              storageClassName:
                type: string
                description: "StorageClass to use"
          backup:
            type: object
            description: "Backup configuration"
            properties:
              enabled:
                type: boolean
                default: true
              schedule:
                type: string
                description: "Cron schedule for backups"
                default: "0 2 * * *"
              retention:
                type: integer
                description: "Number of backups to retain"
                minimum: 1
                default: 7
          connection:
            type: object
            description: "Connection settings"
            properties:
              maxConnections:
                type: integer
                minimum: 10
                maximum: 10000
                default: 100
              sslMode:
                type: string
                enum: ["disable", "require", "verify-ca", "verify-full"]
                default: "require"
    # x-kubernetes-preserve-unknown-fields: true  # Uncomment to allow extra fields
```

### 3.2 Validation Rules (CEL)

Common Expression Language (CEL) rules provide cross-field validation (Kubernetes 1.25+).

```yaml
schema:
  openAPIV3Schema:
    type: object
    properties:
      spec:
        type: object
        x-kubernetes-validations:
          # Cross-field validation: replicas must be odd for postgres
          - rule: "self.engine != 'postgres' || self.replicas % 2 == 1"
            message: "PostgreSQL replicas must be an odd number for quorum"

          # Conditional requirement: if backup enabled, schedule must be set
          - rule: "!has(self.backup) || !self.backup.enabled || has(self.backup.schedule)"
            message: "Backup schedule is required when backups are enabled"

          # Immutable field: engine cannot be changed after creation
          - rule: "self.engine == oldSelf.engine"
            message: "Engine type is immutable and cannot be changed"

          # String format validation
          - rule: "self.version.matches('^[0-9]+\\\\.[0-9]+$')"
            message: "Version must be in format MAJOR.MINOR (e.g., 16.2)"
        properties:
          engine:
            type: string
            enum: ["postgres", "mysql", "mongodb"]
          version:
            type: string
          replicas:
            type: integer
            minimum: 1
            maximum: 10
            x-kubernetes-validations:
              - rule: "self >= oldSelf"
                message: "Replicas can only be scaled up, not down"
          storage:
            type: object
            properties:
              size:
                type: string
                x-kubernetes-validations:
                  - rule: "self == oldSelf"
                    message: "Storage size is immutable (resize not supported)"
```

### 3.3 Default Values

```yaml
# Defaults are applied server-side when fields are omitted
properties:
  spec:
    type: object
    properties:
      replicas:
        type: integer
        default: 1
      storage:
        type: object
        default: {}      # Ensures the object exists so nested defaults apply
        properties:
          size:
            type: string
            default: "10Gi"
          storageClassName:
            type: string
            default: "standard"
      monitoring:
        type: object
        default: {}
        properties:
          enabled:
            type: boolean
            default: true
          interval:
            type: string
            default: "30s"
```

```bash
# Creating a minimal resource -- defaults fill in the rest
cat <<EOF | kubectl apply -f -
apiVersion: example.com/v1alpha1
kind: Database
metadata:
  name: test-db
  namespace: default
spec:
  engine: postgres
  version: "16.2"
EOF

# Verify defaults were applied
kubectl get db test-db -o yaml
# spec:
#   engine: postgres
#   version: "16.2"
#   replicas: 1              <-- default
#   storage:
#     size: 10Gi             <-- default
#     storageClassName: standard  <-- default
#   monitoring:
#     enabled: true           <-- default
#     interval: 30s           <-- default
```

### 3.4 Enum and Pattern Constraints

```yaml
properties:
  spec:
    type: object
    properties:
      # Enum: fixed set of allowed values
      tier:
        type: string
        enum: ["development", "staging", "production"]
        description: "Service tier determines resource allocation"

      # Pattern: regex validation
      name:
        type: string
        pattern: "^[a-z][a-z0-9-]{0,61}[a-z0-9]$"
        description: "DNS-compatible name"
        minLength: 2
        maxLength: 63

      # Numeric constraints
      port:
        type: integer
        minimum: 1024
        maximum: 65535
        exclusiveMinimum: true   # port > 1024

      # Array constraints
      allowedCIDRs:
        type: array
        items:
          type: string
          pattern: "^[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}/[0-9]{1,2}$"
        minItems: 1
        maxItems: 50

      # Map/object with additional properties
      labels:
        type: object
        additionalProperties:
          type: string
        x-kubernetes-map-type: granular   # Allow individual field updates
```

---

## 4. CRD Versioning

### 4.1 Multiple Versions

CRDs can serve multiple versions simultaneously, allowing gradual migration.

```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: databases.example.com
spec:
  group: example.com
  names:
    kind: Database
    plural: databases
    singular: database
    shortNames: ["db"]
  scope: Namespaced
  versions:
    - name: v1alpha1
      served: true              # Still accepts requests
      storage: false            # NOT the storage version
      deprecated: true          # Mark as deprecated
      deprecationWarning: "example.com/v1alpha1 Database is deprecated; use v1beta1"
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              properties:
                engine:
                  type: string
                version:
                  type: string
                replicas:
                  type: integer
    - name: v1beta1
      served: true
      storage: true             # This is the storage version
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              required:
                - engine
                - version
              properties:
                engine:
                  type: string
                  enum: ["postgres", "mysql", "mongodb"]
                version:
                  type: string
                replicas:
                  type: integer
                  minimum: 1
                  maximum: 10
                  default: 1
                # New fields in v1beta1
                highAvailability:
                  type: object
                  properties:
                    enabled:
                      type: boolean
                      default: false
                    syncMode:
                      type: string
                      enum: ["sync", "async"]
                      default: "async"
```

### 4.2 Storage Version

The storage version is the version used to persist objects in etcd. Only one version can be the storage version at a time.

```bash
# Check which version is the storage version
kubectl get crd databases.example.com -o jsonpath='{.status.storedVersions}'
# ["v1beta1"]

# When migrating storage versions:
# 1. Add new version with storage: true
# 2. Set old version to storage: false
# 3. Migrate existing objects: read and re-write each object
# 4. Remove old version from storedVersions
kubectl get db --all-namespaces -o json | kubectl replace -f -

# Verify migration
kubectl get crd databases.example.com -o jsonpath='{.status.storedVersions}'
# ["v1beta1"]
```

### 4.3 Conversion Webhooks

Conversion webhooks translate between CRD versions when a client requests a version different from the storage version.

```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: databases.example.com
spec:
  group: example.com
  conversion:
    strategy: Webhook
    webhook:
      clientConfig:
        service:
          name: database-conversion-webhook
          namespace: system
          path: /convert
          port: 443
        caBundle: LS0tLS1CRUdJTi...    # Base64-encoded CA certificate
      conversionReviewVersions:
        - v1
  # ... versions, names, scope
```

A conversion webhook implementation in Go:

```go
package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

func handleConvert(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	var review apiextensionsv1.ConversionReview
	if err := json.Unmarshal(body, &review); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	desiredVersion := review.Request.DesiredAPIVersion
	convertedObjects := make([]unstructured.Unstructured, 0, len(review.Request.Objects))

	for _, raw := range review.Request.Objects {
		var obj unstructured.Unstructured
		if err := json.Unmarshal(raw.Raw, &obj); err != nil {
			sendError(w, &review, fmt.Sprintf("failed to unmarshal: %v", err))
			return
		}

		// Convert based on source and target versions
		converted, err := convert(&obj, desiredVersion)
		if err != nil {
			sendError(w, &review, fmt.Sprintf("conversion failed: %v", err))
			return
		}
		convertedObjects = append(convertedObjects, *converted)
	}

	review.Response = &apiextensionsv1.ConversionResponse{
		UID:              review.Request.UID,
		Result:           successStatus(),
		ConvertedObjects: toRawExtensions(convertedObjects),
	}
	review.Request = nil

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(review)
}

func convert(obj *unstructured.Unstructured, targetVersion string) (*unstructured.Unstructured, error) {
	sourceVersion := obj.GetAPIVersion()

	// v1alpha1 -> v1beta1: add highAvailability defaults
	if sourceVersion == "example.com/v1alpha1" && targetVersion == "example.com/v1beta1" {
		spec, _, _ := unstructured.NestedMap(obj.Object, "spec")
		if _, exists := spec["highAvailability"]; !exists {
			spec["highAvailability"] = map[string]interface{}{
				"enabled":  false,
				"syncMode": "async",
			}
		}
		unstructured.SetNestedMap(obj.Object, spec, "spec")
		obj.SetAPIVersion(targetVersion)
		return obj, nil
	}

	// v1beta1 -> v1alpha1: strip highAvailability
	if sourceVersion == "example.com/v1beta1" && targetVersion == "example.com/v1alpha1" {
		spec, _, _ := unstructured.NestedMap(obj.Object, "spec")
		delete(spec, "highAvailability")
		unstructured.SetNestedMap(obj.Object, spec, "spec")
		obj.SetAPIVersion(targetVersion)
		return obj, nil
	}

	return obj, nil
}

func main() {
	http.HandleFunc("/convert", handleConvert)
	http.ListenAndServeTLS(":8443", "/certs/tls.crt", "/certs/tls.key", nil)
}
```

---

## 5. Subresources

### 5.1 Status Subresource

The status subresource separates spec (desired state, user-writable) from status (observed state, controller-writable). With the status subresource enabled, `kubectl apply` cannot modify `.status`, and status updates cannot modify `.spec`.

```yaml
# In the CRD version definition:
versions:
  - name: v1beta1
    served: true
    storage: true
    subresources:
      status: {}              # Enable the status subresource
    schema:
      openAPIV3Schema:
        type: object
        properties:
          spec:
            type: object
            properties:
              engine:
                type: string
              replicas:
                type: integer
          status:
            type: object
            properties:
              phase:
                type: string
                enum: ["Creating", "Running", "Failed", "Deleting"]
              readyReplicas:
                type: integer
              observedGeneration:
                type: integer
                format: int64
              conditions:
                type: array
                items:
                  type: object
                  required: ["type", "status"]
                  properties:
                    type:
                      type: string
                    status:
                      type: string
                      enum: ["True", "False", "Unknown"]
                    lastTransitionTime:
                      type: string
                      format: date-time
                    reason:
                      type: string
                    message:
                      type: string
```

```bash
# Update status from a controller (uses the /status subresource endpoint)
kubectl proxy &

curl -X PUT http://localhost:8001/apis/example.com/v1beta1/namespaces/default/databases/orders-db/status \
  -H "Content-Type: application/json" \
  -d '{
    "apiVersion": "example.com/v1beta1",
    "kind": "Database",
    "metadata": {
      "name": "orders-db",
      "namespace": "default"
    },
    "status": {
      "phase": "Running",
      "readyReplicas": 3,
      "observedGeneration": 1,
      "conditions": [
        {
          "type": "Ready",
          "status": "True",
          "lastTransitionTime": "2024-01-15T10:05:00Z",
          "reason": "AllReplicasReady",
          "message": "All 3 replicas are ready"
        }
      ]
    }
  }'
```

Updating status from Go (the typical controller pattern):

```go
package main

import (
	"context"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/rest"
)

func updateDatabaseStatus(client dynamic.Interface, name, namespace string) error {
	gvr := schema.GroupVersionResource{
		Group:    "example.com",
		Version:  "v1beta1",
		Resource: "databases",
	}

	// Get the current resource
	db, err := client.Resource(gvr).Namespace(namespace).Get(
		context.TODO(), name, metav1.GetOptions{},
	)
	if err != nil {
		return err
	}

	// Update status fields
	status := map[string]interface{}{
		"phase":              "Running",
		"readyReplicas":      int64(3),
		"observedGeneration": db.GetGeneration(),
		"conditions": []interface{}{
			map[string]interface{}{
				"type":               "Ready",
				"status":             "True",
				"lastTransitionTime": time.Now().UTC().Format(time.RFC3339),
				"reason":             "AllReplicasReady",
				"message":            "All replicas are ready",
			},
		},
	}
	unstructured.SetNestedMap(db.Object, status, "status")

	// Update via the /status subresource
	_, err = client.Resource(gvr).Namespace(namespace).UpdateStatus(
		context.TODO(), db, metav1.UpdateOptions{},
	)
	return err
}

func main() {
	config, _ := rest.InClusterConfig()
	client, _ := dynamic.NewForConfig(config)
	updateDatabaseStatus(client, "orders-db", "default")
}
```

### 5.2 Scale Subresource

The scale subresource enables `kubectl scale` and HPA integration for your custom resource.

```yaml
versions:
  - name: v1beta1
    served: true
    storage: true
    subresources:
      status: {}
      scale:
        specReplicasPath: .spec.replicas
        statusReplicasPath: .status.readyReplicas
        labelSelectorPath: .status.selector    # Optional: for HPA
    schema:
      openAPIV3Schema:
        type: object
        properties:
          spec:
            type: object
            properties:
              replicas:
                type: integer
                minimum: 1
                maximum: 10
          status:
            type: object
            properties:
              readyReplicas:
                type: integer
              selector:
                type: string
```

```bash
# Now kubectl scale works with your custom resource
kubectl scale db orders-db --replicas=5

# HPA can also target your custom resource
kubectl autoscale db orders-db --min=3 --max=10 --cpu-percent=80
```

```yaml
# HPA targeting a custom resource
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: orders-db-hpa
spec:
  scaleTargetRef:
    apiVersion: example.com/v1beta1
    kind: Database
    name: orders-db
  minReplicas: 3
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

---

## 6. Printer Columns

Printer columns define what `kubectl get` shows in the table output.

```yaml
versions:
  - name: v1beta1
    served: true
    storage: true
    additionalPrinterColumns:
      - name: Engine
        type: string
        jsonPath: .spec.engine
        description: "Database engine type"
      - name: Version
        type: string
        jsonPath: .spec.version
        description: "Engine version"
      - name: Replicas
        type: integer
        jsonPath: .spec.replicas
        description: "Desired number of replicas"
      - name: Ready
        type: integer
        jsonPath: .status.readyReplicas
        description: "Number of ready replicas"
      - name: Phase
        type: string
        jsonPath: .status.phase
        description: "Current phase"
      - name: Storage
        type: string
        jsonPath: .spec.storage.size
        priority: 1              # Only shown with -o wide
        description: "Storage size"
      - name: Age
        type: date
        jsonPath: .metadata.creationTimestamp
```

```bash
# Default view
kubectl get db -n production
# NAME        ENGINE     VERSION   REPLICAS   READY   PHASE     AGE
# orders-db   postgres   16.2      3          3       Running   2d
# users-db    postgres   16.2      1          1       Running   5d

# Wide view (includes priority 1 columns)
kubectl get db -n production -o wide
# NAME        ENGINE     VERSION   REPLICAS   READY   PHASE     STORAGE   AGE
# orders-db   postgres   16.2      3          3       Running   100Gi     2d
# users-db    postgres   16.2      1          1       Running   10Gi      5d
```

---

## 7. Categories and Short Names

```yaml
names:
  kind: Database
  listKind: DatabaseList
  singular: database
  plural: databases
  shortNames:
    - db                # kubectl get db
    - dbs               # kubectl get dbs
  categories:
    - all               # Included in kubectl get all
    - example           # kubectl get example (shows all resources in this category)
    - databases         # kubectl get databases (custom category)
```

```bash
# Short names
kubectl get db                  # same as kubectl get databases
kubectl get dbs                 # same as kubectl get databases

# Categories
kubectl get all -n production   # Now includes Database resources
kubectl get example             # Shows all resources with category "example"

# Check available short names
kubectl api-resources | grep database
# databases   db,dbs   example.com/v1beta1   true   Database
```

---

## 8. CRD Best Practices

### 8.1 Design Guidelines

| Guideline | Rationale |
|-----------|-----------|
| Use a domain you own for the API group | Prevents conflicts (e.g., `example.com`, not `k8s.io`) |
| Start with `v1alpha1` | Communicate instability; promote as API matures |
| Separate spec from status | Use status subresource; controllers update status |
| Make spec fields declarative | Describe desired state, not imperative actions |
| Use conditions for status | Follow the standard `type`/`status`/`reason`/`message` pattern |
| Set `observedGeneration` in status | Controllers should report the last spec generation they processed |
| Include printer columns | Better `kubectl get` experience |
| Add validation rules | Catch errors early; use CEL for cross-field validation |
| Document all fields | Use `description` in the schema |
| Use finalizers for cleanup | Prevent deletion until external resources are cleaned up |

### 8.2 Schema Evolution

```
v1alpha1 ──▶ v1alpha2 ──▶ v1beta1 ──▶ v1
   │              │            │         │
   │  Breaking    │  Breaking  │  No     │  Stable
   │  changes     │  changes   │  breaking│  API
   │  allowed     │  allowed   │  changes │
   │              │            │         │
   │  Short       │  Longer    │  Long    │  Indefinite
   │  support     │  support   │  support │  support
```

Rules for each stability level:

- **v1alpha1/v1alpha2**: Breaking changes allowed. May be dropped without migration path.
- **v1beta1**: Breaking changes discouraged. Migration path should be provided.
- **v1**: No breaking changes. Fields can be added but not removed or renamed.

### 8.3 Performance Considerations

```yaml
# Large CRDs: set x-kubernetes-list-type for efficient updates
properties:
  spec:
    type: object
    properties:
      endpoints:
        type: array
        x-kubernetes-list-type: map       # Merge by key, not replace entire list
        x-kubernetes-list-map-keys:
          - name
        items:
          type: object
          required: ["name"]
          properties:
            name:
              type: string
            address:
              type: string
            port:
              type: integer
```

```bash
# Monitor CRD storage usage
kubectl get --raw /metrics | grep apiserver_storage_objects
# apiserver_storage_objects{resource="databases.example.com"} 150

# Watch for excessive object size
kubectl get db -A -o json | wc -c
# Keep individual objects under 1.5MB (etcd limit)
```

---

## 9. Aggregated API Servers vs CRDs

### 9.1 When to Use Each

| Factor | CRDs | Aggregated API Servers |
|--------|------|----------------------|
| **Implementation effort** | YAML manifest only | Full Go API server |
| **Storage** | etcd (via API server) | Custom (any backend) |
| **Validation** | OpenAPI schema + CEL | Arbitrary Go code |
| **Authentication** | Kubernetes-native | Custom or Kubernetes |
| **Subresources** | status, scale only | Any subresource |
| **API discovery** | Automatic | Via APIService registration |
| **Long-running requests** | Not supported | Supported (watch, exec) |
| **Protobuf support** | No (JSON only) | Yes |
| **Use cases** | Most custom resources | metrics-server, custom-metrics |

**Choose CRDs** when:
- Your data fits the Kubernetes resource model (metadata, spec, status)
- You need standard CRUD operations
- OpenAPI validation + CEL is sufficient
- You want minimal operational overhead

**Choose Aggregated API Servers** when:
- You need a custom storage backend (not etcd)
- You need subresources beyond status and scale
- You need protobuf for performance
- You need streaming or long-running requests

### 9.2 Aggregated API Server Example

```yaml
# Register an aggregated API server with Kubernetes
apiVersion: apiregistration.k8s.io/v1
kind: APIService
metadata:
  name: v1beta1.custom.metrics.k8s.io
spec:
  service:
    name: custom-metrics-apiserver
    namespace: monitoring
    port: 443
  group: custom.metrics.k8s.io
  version: v1beta1
  insecureSkipTLSVerify: false
  caBundle: LS0tLS1CRUdJTi...
  groupPriorityMinimum: 100
  versionPriority: 100
```

```bash
# The metrics-server is a real-world example of an aggregated API server
kubectl get apiservices | grep metrics
# v1beta1.metrics.k8s.io   kube-system/metrics-server   True   30d

# It serves custom API endpoints
kubectl get --raw /apis/metrics.k8s.io/v1beta1/nodes
kubectl top nodes   # Uses the aggregated API behind the scenes
```

---

## Exercises

### Exercise 1: Create a CRD for a Cache Resource

Create a CRD for a `Cache` resource in the `infrastructure.example.com` group with the following fields:
- `engine` (required, enum: redis, memcached)
- `version` (required, string)
- `replicas` (default: 1, min: 1, max: 5)
- `memory` (string, pattern for sizes like "256Mi", "1Gi")
- `evictionPolicy` (enum: noeviction, allkeys-lru, volatile-lru)

Include printer columns for engine, version, replicas, and age.

<details><summary>Show Answer</summary>

```yaml
# cache-crd.yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: caches.infrastructure.example.com
spec:
  group: infrastructure.example.com
  names:
    kind: Cache
    listKind: CacheList
    singular: cache
    plural: caches
    shortNames:
      - ca
    categories:
      - all
      - infrastructure
  scope: Namespaced
  versions:
    - name: v1alpha1
      served: true
      storage: true
      subresources:
        status: {}
      additionalPrinterColumns:
        - name: Engine
          type: string
          jsonPath: .spec.engine
        - name: Version
          type: string
          jsonPath: .spec.version
        - name: Replicas
          type: integer
          jsonPath: .spec.replicas
        - name: Memory
          type: string
          jsonPath: .spec.memory
          priority: 1
        - name: Age
          type: date
          jsonPath: .metadata.creationTimestamp
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              required:
                - engine
                - version
              properties:
                engine:
                  type: string
                  enum: ["redis", "memcached"]
                version:
                  type: string
                replicas:
                  type: integer
                  minimum: 1
                  maximum: 5
                  default: 1
                memory:
                  type: string
                  pattern: "^[0-9]+(Mi|Gi)$"
                  default: "256Mi"
                evictionPolicy:
                  type: string
                  enum: ["noeviction", "allkeys-lru", "volatile-lru"]
                  default: "allkeys-lru"
            status:
              type: object
              properties:
                phase:
                  type: string
                readyReplicas:
                  type: integer
```

```bash
kubectl apply -f cache-crd.yaml

# Create a cache instance
cat <<EOF | kubectl apply -f -
apiVersion: infrastructure.example.com/v1alpha1
kind: Cache
metadata:
  name: session-cache
  namespace: default
spec:
  engine: redis
  version: "7.2"
  replicas: 3
  memory: "1Gi"
  evictionPolicy: volatile-lru
EOF

# Verify
kubectl get caches
# NAME            ENGINE   VERSION   REPLICAS   AGE
# session-cache   redis    7.2       3          5s

kubectl get ca -o wide
# NAME            ENGINE   VERSION   REPLICAS   MEMORY   AGE
# session-cache   redis    7.2       3          1Gi      10s

# Test validation
cat <<EOF | kubectl apply -f -
apiVersion: infrastructure.example.com/v1alpha1
kind: Cache
metadata:
  name: bad-cache
spec:
  engine: couchbase
  version: "1.0"
EOF
# Error: spec.engine: Unsupported value: "couchbase": supported values: "redis", "memcached"
```

</details>

### Exercise 2: Add CEL Validation Rules

Extend the Cache CRD from Exercise 1 with CEL validation rules:
- Engine is immutable after creation
- If engine is "memcached", replicas must be exactly 1 (memcached does not support clustering natively)
- Memory must be at least "128Mi" for redis and "64Mi" for memcached

<details><summary>Show Answer</summary>

```yaml
# cache-crd-validated.yaml (relevant spec.versions section)
versions:
  - name: v1alpha1
    served: true
    storage: true
    subresources:
      status: {}
    additionalPrinterColumns:
      - name: Engine
        type: string
        jsonPath: .spec.engine
      - name: Version
        type: string
        jsonPath: .spec.version
      - name: Replicas
        type: integer
        jsonPath: .spec.replicas
      - name: Age
        type: date
        jsonPath: .metadata.creationTimestamp
    schema:
      openAPIV3Schema:
        type: object
        properties:
          spec:
            type: object
            required:
              - engine
              - version
            x-kubernetes-validations:
              # Engine is immutable
              - rule: "self.engine == oldSelf.engine"
                message: "Engine type cannot be changed after creation"
              # Memcached: replicas must be 1
              - rule: "self.engine != 'memcached' || self.replicas == 1"
                message: "Memcached does not support clustering; replicas must be 1"
              # Memory minimum based on engine
              - rule: >-
                  self.engine != 'redis' ||
                  (self.memory.endsWith('Gi') ||
                   (self.memory.endsWith('Mi') &&
                    int(self.memory.replace('Mi','')) >= 128))
                message: "Redis requires at least 128Mi of memory"
              - rule: >-
                  self.engine != 'memcached' ||
                  (self.memory.endsWith('Gi') ||
                   (self.memory.endsWith('Mi') &&
                    int(self.memory.replace('Mi','')) >= 64))
                message: "Memcached requires at least 64Mi of memory"
            properties:
              engine:
                type: string
                enum: ["redis", "memcached"]
              version:
                type: string
              replicas:
                type: integer
                minimum: 1
                maximum: 5
                default: 1
              memory:
                type: string
                pattern: "^[0-9]+(Mi|Gi)$"
                default: "256Mi"
              evictionPolicy:
                type: string
                enum: ["noeviction", "allkeys-lru", "volatile-lru"]
                default: "allkeys-lru"
          status:
            type: object
            properties:
              phase:
                type: string
              readyReplicas:
                type: integer
```

```bash
kubectl apply -f cache-crd-validated.yaml

# Test: memcached with replicas > 1 should fail
cat <<EOF | kubectl apply -f -
apiVersion: infrastructure.example.com/v1alpha1
kind: Cache
metadata:
  name: bad-memcached
spec:
  engine: memcached
  version: "1.6"
  replicas: 3
EOF
# Error: Memcached does not support clustering; replicas must be 1

# Test: redis with insufficient memory should fail
cat <<EOF | kubectl apply -f -
apiVersion: infrastructure.example.com/v1alpha1
kind: Cache
metadata:
  name: small-redis
spec:
  engine: redis
  version: "7.2"
  memory: "64Mi"
EOF
# Error: Redis requires at least 128Mi of memory

# Test: changing engine should fail
kubectl patch cache session-cache --type=merge -p '{"spec":{"engine":"memcached"}}'
# Error: Engine type cannot be changed after creation
```

</details>

### Exercise 3: CRD with Status Subresource

Create a CRD for a `Backup` resource that includes a status subresource with conditions. Write a shell script that simulates a controller updating the status through the API.

<details><summary>Show Answer</summary>

```yaml
# backup-crd.yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: backups.storage.example.com
spec:
  group: storage.example.com
  names:
    kind: Backup
    plural: backups
    singular: backup
    shortNames: ["bk"]
  scope: Namespaced
  versions:
    - name: v1alpha1
      served: true
      storage: true
      subresources:
        status: {}
      additionalPrinterColumns:
        - name: Source
          type: string
          jsonPath: .spec.source
        - name: Phase
          type: string
          jsonPath: .status.phase
        - name: Size
          type: string
          jsonPath: .status.backupSize
        - name: Age
          type: date
          jsonPath: .metadata.creationTimestamp
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              required: ["source", "destination"]
              properties:
                source:
                  type: string
                destination:
                  type: string
                schedule:
                  type: string
            status:
              type: object
              properties:
                phase:
                  type: string
                  enum: ["Pending", "InProgress", "Completed", "Failed"]
                backupSize:
                  type: string
                startedAt:
                  type: string
                  format: date-time
                completedAt:
                  type: string
                  format: date-time
                conditions:
                  type: array
                  items:
                    type: object
                    required: ["type", "status"]
                    properties:
                      type:
                        type: string
                      status:
                        type: string
                      lastTransitionTime:
                        type: string
                        format: date-time
                      reason:
                        type: string
                      message:
                        type: string
```

```bash
kubectl apply -f backup-crd.yaml

# Create a backup resource
cat <<EOF | kubectl apply -f -
apiVersion: storage.example.com/v1alpha1
kind: Backup
metadata:
  name: daily-backup
  namespace: default
spec:
  source: orders-db
  destination: s3://my-bucket/backups
  schedule: "0 2 * * *"
EOF

# Simulate controller updating status
# Start kubectl proxy in the background
kubectl proxy --port=8001 &
PROXY_PID=$!

# Get the current resource for the resourceVersion
RESOURCE=$(curl -s http://localhost:8001/apis/storage.example.com/v1alpha1/namespaces/default/backups/daily-backup)
RV=$(echo "$RESOURCE" | jq -r '.metadata.resourceVersion')

# Update status to InProgress
curl -s -X PUT \
  http://localhost:8001/apis/storage.example.com/v1alpha1/namespaces/default/backups/daily-backup/status \
  -H "Content-Type: application/json" \
  -d "{
    \"apiVersion\": \"storage.example.com/v1alpha1\",
    \"kind\": \"Backup\",
    \"metadata\": {
      \"name\": \"daily-backup\",
      \"namespace\": \"default\",
      \"resourceVersion\": \"$RV\"
    },
    \"status\": {
      \"phase\": \"InProgress\",
      \"startedAt\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
      \"conditions\": [
        {
          \"type\": \"Started\",
          \"status\": \"True\",
          \"lastTransitionTime\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
          \"reason\": \"BackupStarted\",
          \"message\": \"Backup process has started\"
        }
      ]
    }
  }" | jq .

# Verify status
kubectl get bk daily-backup
# NAME           SOURCE      PHASE        SIZE   AGE
# daily-backup   orders-db   InProgress          30s

# Clean up proxy
kill $PROXY_PID
```

</details>

### Exercise 4: Multi-Version CRD

Create a CRD with two versions (`v1alpha1` and `v1beta1`). The `v1beta1` version adds a `monitoring` field that does not exist in `v1alpha1`. Create resources using both versions and verify they are accessible from either version.

<details><summary>Show Answer</summary>

```yaml
# multi-version-crd.yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: appconfigs.config.example.com
spec:
  group: config.example.com
  names:
    kind: AppConfig
    plural: appconfigs
    singular: appconfig
    shortNames: ["ac"]
  scope: Namespaced
  versions:
    - name: v1alpha1
      served: true
      storage: false
      deprecated: true
      deprecationWarning: "config.example.com/v1alpha1 AppConfig is deprecated; migrate to v1beta1"
      additionalPrinterColumns:
        - name: Port
          type: integer
          jsonPath: .spec.port
        - name: Age
          type: date
          jsonPath: .metadata.creationTimestamp
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              required: ["appName", "port"]
              properties:
                appName:
                  type: string
                port:
                  type: integer
                replicas:
                  type: integer
                  default: 1
            status:
              type: object
              properties:
                phase:
                  type: string
    - name: v1beta1
      served: true
      storage: true
      additionalPrinterColumns:
        - name: Port
          type: integer
          jsonPath: .spec.port
        - name: Monitoring
          type: boolean
          jsonPath: .spec.monitoring.enabled
        - name: Age
          type: date
          jsonPath: .metadata.creationTimestamp
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              required: ["appName", "port"]
              properties:
                appName:
                  type: string
                port:
                  type: integer
                replicas:
                  type: integer
                  default: 1
                monitoring:
                  type: object
                  default: {}
                  properties:
                    enabled:
                      type: boolean
                      default: false
                    metricsPath:
                      type: string
                      default: "/metrics"
                    interval:
                      type: string
                      default: "30s"
            status:
              type: object
              properties:
                phase:
                  type: string
```

```bash
kubectl apply -f multi-version-crd.yaml

# Create a resource using v1alpha1 (deprecated)
cat <<EOF | kubectl apply -f -
apiVersion: config.example.com/v1alpha1
kind: AppConfig
metadata:
  name: legacy-app
spec:
  appName: legacy-service
  port: 8080
  replicas: 2
EOF
# Warning: config.example.com/v1alpha1 AppConfig is deprecated; migrate to v1beta1

# Create a resource using v1beta1
cat <<EOF | kubectl apply -f -
apiVersion: config.example.com/v1beta1
kind: AppConfig
metadata:
  name: modern-app
spec:
  appName: modern-service
  port: 9090
  replicas: 3
  monitoring:
    enabled: true
    metricsPath: "/api/metrics"
    interval: "15s"
EOF

# Access v1alpha1-created resource via v1beta1 API
kubectl get appconfigs.v1beta1.config.example.com legacy-app -o yaml
# The monitoring field will have defaults applied

# Access v1beta1-created resource via v1alpha1 API
kubectl get appconfigs.v1alpha1.config.example.com modern-app -o yaml
# The monitoring field is present but not in v1alpha1 schema

# List all AppConfigs (shows both)
kubectl get ac
# NAME          PORT   MONITORING   AGE
# legacy-app    8080   false        30s
# modern-app    9090   true         15s
```

</details>

### Exercise 5: CRD with Scale Subresource

Create a CRD for a `WorkerPool` resource that supports the scale subresource. Verify you can use `kubectl scale` and that an HPA can target it.

<details><summary>Show Answer</summary>

```yaml
# workerpool-crd.yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: workerpools.compute.example.com
spec:
  group: compute.example.com
  names:
    kind: WorkerPool
    plural: workerpools
    singular: workerpool
    shortNames: ["wp"]
    categories:
      - all
  scope: Namespaced
  versions:
    - name: v1alpha1
      served: true
      storage: true
      subresources:
        status: {}
        scale:
          specReplicasPath: .spec.workers
          statusReplicasPath: .status.readyWorkers
          labelSelectorPath: .status.labelSelector
      additionalPrinterColumns:
        - name: Workers
          type: integer
          jsonPath: .spec.workers
        - name: Ready
          type: integer
          jsonPath: .status.readyWorkers
        - name: Task
          type: string
          jsonPath: .spec.taskType
        - name: Age
          type: date
          jsonPath: .metadata.creationTimestamp
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              required: ["taskType", "workers"]
              properties:
                taskType:
                  type: string
                  enum: ["batch", "stream", "ml-training"]
                workers:
                  type: integer
                  minimum: 0
                  maximum: 100
                workerTemplate:
                  type: object
                  properties:
                    image:
                      type: string
                    resources:
                      type: object
                      properties:
                        cpu:
                          type: string
                        memory:
                          type: string
            status:
              type: object
              properties:
                readyWorkers:
                  type: integer
                labelSelector:
                  type: string
                conditions:
                  type: array
                  items:
                    type: object
                    required: ["type", "status"]
                    properties:
                      type:
                        type: string
                      status:
                        type: string
```

```bash
kubectl apply -f workerpool-crd.yaml

# Create a WorkerPool
cat <<EOF | kubectl apply -f -
apiVersion: compute.example.com/v1alpha1
kind: WorkerPool
metadata:
  name: data-processors
  namespace: default
spec:
  taskType: batch
  workers: 5
  workerTemplate:
    image: my-worker:v1
    resources:
      cpu: "500m"
      memory: "1Gi"
EOF

# Scale using kubectl
kubectl scale wp data-processors --replicas=10
kubectl get wp data-processors
# NAME              WORKERS   READY   TASK    AGE
# data-processors   10                batch   30s

# Verify the scale subresource works
kubectl get --raw /apis/compute.example.com/v1alpha1/namespaces/default/workerpools/data-processors/scale | jq .
# {
#   "kind": "Scale",
#   "apiVersion": "autoscaling/v1",
#   "metadata": { "name": "data-processors", ... },
#   "spec": { "replicas": 10 },
#   "status": { "replicas": 0 }
# }

# Create an HPA targeting the WorkerPool
cat <<EOF | kubectl apply -f -
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: data-processors-hpa
spec:
  scaleTargetRef:
    apiVersion: compute.example.com/v1alpha1
    kind: WorkerPool
    name: data-processors
  minReplicas: 3
  maxReplicas: 50
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 75
EOF

kubectl get hpa data-processors-hpa
# NAME                    REFERENCE                    TARGETS   MINPODS   MAXPODS
# data-processors-hpa     WorkerPool/data-processors   <unknown> 3         50
```

</details>

---

**Previous**: [Helm and Kustomize](./09_Helm_and_Kustomize.md) | **Next**: [Operators](./11_Operators.md)
