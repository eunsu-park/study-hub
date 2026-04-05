#!/bin/bash
# Exercise: Lesson 09 — Helm and Kustomize
# Complete the TODO items below.

# === Exercise 1: Create a Helm Chart from Scratch ===
# Build a basic Helm chart for a web application.
# Hint: 'helm create' scaffolds a chart with sensible defaults.

exercise_1() {
    echo "=== Exercise 1: Create a Helm Chart ==="

    # TODO: Create a new Helm chart named "webapp"
    # Command: helm create webapp

    # TODO: Edit webapp/values.yaml to customize:
    #   image.repository: nginx
    #   image.tag: "1.25-alpine"
    #   service.type: NodePort
    #   service.port: 80
    #   replicaCount: 3

    # TODO: Template the chart locally and review the output
    # Command: helm template my-release ./webapp

    # TODO: Install the chart into the cluster
    # Command: helm install my-release ./webapp --namespace default

    # TODO: Verify the deployment
    # Command: helm list
    # Command: kubectl get all -l app.kubernetes.io/instance=my-release

}

# === Exercise 2: Helm Values Override ===
# Deploy the same chart with different configs for dev and prod.
# Hint: Use -f to specify a values file, --set for individual overrides.

exercise_2() {
    echo "=== Exercise 2: Helm Values Override ==="

    # TODO: Create a file values-dev.yaml with:
    #   replicaCount: 1
    #   resources:
    #     limits:
    #       cpu: 100m
    #       memory: 128Mi

    # TODO: Create a file values-prod.yaml with:
    #   replicaCount: 5
    #   resources:
    #     limits:
    #       cpu: 500m
    #       memory: 512Mi

    # TODO: Install with dev values
    # Command: helm install dev-release ./webapp -f values-dev.yaml -n dev

    # TODO: Install with prod values and an additional override
    # Command: helm install prod-release ./webapp -f values-prod.yaml \
    #          --set ingress.enabled=true -n prod

    # TODO: Compare the two releases
    # Command: helm get values dev-release -n dev
    # Command: helm get values prod-release -n prod

}

# === Exercise 3: Helm Upgrade and Rollback ===
# Practice upgrading and rolling back a Helm release.
# Hint: Helm keeps release history for rollback support.

exercise_3() {
    echo "=== Exercise 3: Helm Upgrade and Rollback ==="

    # TODO: Upgrade the release with a new image tag
    # Command: helm upgrade my-release ./webapp --set image.tag="1.26-alpine"

    # TODO: Check the release history
    # Command: helm history my-release

    # TODO: Roll back to the previous revision
    # Command: helm rollback my-release 1

    # TODO: Verify the rollback succeeded
    # Command: helm status my-release
    # Command: kubectl get pods -l app.kubernetes.io/instance=my-release \
    #          -o jsonpath='{.items[*].spec.containers[*].image}'

}

# === Exercise 4: Kustomize Base and Overlay ===
# Use Kustomize to manage environment-specific configurations.
# Hint: Kustomize uses patches — no templating language needed.

exercise_4() {
    echo "=== Exercise 4: Kustomize Overlay ==="

    # TODO: Create the directory structure:
    #   kustomize-demo/
    #   ├── base/
    #   │   ├── kustomization.yaml
    #   │   ├── deployment.yaml
    #   │   └── service.yaml
    #   └── overlays/
    #       ├── dev/
    #       │   └── kustomization.yaml
    #       └── prod/
    #           ├── kustomization.yaml
    #           └── replica-patch.yaml

    # TODO: In base/kustomization.yaml, list the resources:
    #   resources:
    #   - deployment.yaml
    #   - service.yaml

    # TODO: In overlays/dev/kustomization.yaml, set:
    #   resources:
    #   - ../../base
    #   namePrefix: dev-
    #   namespace: dev
    #   patches:
    #   - target:
    #       kind: Deployment
    #       name: webapp
    #     patch: |
    #       - op: replace
    #         path: /spec/replicas
    #         value: 1

    # TODO: In overlays/prod/kustomization.yaml, set:
    #   resources:
    #   - ../../base
    #   namePrefix: prod-
    #   namespace: prod

    # TODO: Build and review the dev overlay
    # Command: kubectl kustomize overlays/dev/

    # TODO: Apply the prod overlay
    # Command: kubectl apply -k overlays/prod/

}

# === Exercise 5: Combine Helm and Kustomize ===
# Use Kustomize to post-process Helm chart output.
# Hint: helmCharts generator in kustomization.yaml (Kustomize v4.1+).

exercise_5() {
    echo "=== Exercise 5: Helm + Kustomize ==="

    # TODO: Create a kustomization.yaml that uses a Helm chart as input:
    #   helmCharts:
    #   - name: nginx-ingress
    #     repo: https://kubernetes.github.io/ingress-nginx
    #     version: 4.8.0
    #     releaseName: ingress
    #     namespace: ingress-system
    #     valuesFile: values.yaml

    # TODO: Add Kustomize patches on top of the Helm output:
    #   patches:
    #   - target:
    #       kind: Deployment
    #     patch: |
    #       - op: add
    #         path: /metadata/labels/managed-by
    #         value: kustomize

    # TODO: Build and review the combined output
    # Command: kubectl kustomize --enable-helm .

    # TODO: Apply to the cluster
    # Command: kubectl apply -k . --enable-helm

}

# Run all exercises
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
echo ""
exercise_5
echo ""
echo "All exercises completed!"
