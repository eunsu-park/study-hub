# 12. 클라우드 IoT 통합

**이전**: [이미지 분석 프로젝트](./11_Image_Analysis_Project.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. AWS IoT Core 아키텍처와 핵심 구성 요소를 설명할 수 있다
2. AWS IoT Core에서 인증서를 사용하여 IoT 장치를 등록할 수 있다
3. Raspberry Pi에서 GCP Cloud Pub/Sub으로 센서 데이터를 발행할 수 있다
4. 로컬 메시지를 클라우드 서비스로 전달하는 MQTT 브릿지를 구축할 수 있다
5. 클라우드와 장치 간 양방향 통신을 구현할 수 있다
6. InfluxDB와 대시보드 API를 사용하여 IoT 시계열 데이터를 저장하고 분석할 수 있다

---

엣지 처리는 실시간 결정을 처리하지만, 클라우드 통합은 IoT 데이터가 대규모로 진정한 가치를 발휘하는 곳입니다. 클라우드 서비스는 무제한 스토리지, 강력한 분석, 수천 개의 장치에 대한 플릿(fleet) 관리, 그리고 머신러닝 파이프라인과의 통합을 제공합니다. 이 마지막 레슨은 로컬 IoT 시스템을 클라우드에 연결하여 센서부터 인사이트까지 전체 데이터 파이프라인을 완성합니다.

---

## 1. AWS IoT Core 개요

### 1.1 AWS IoT Core 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    AWS IoT Core 아키텍처                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   IoT 디바이스                    AWS Cloud                  │
│   ┌─────────┐                                                │
│   │라즈베리 │     MQTT/TLS      ┌──────────────────────┐    │
│   │  파이   │───────────────────▶│    AWS IoT Core     │    │
│   └─────────┘                    │                      │    │
│                                  │  • Message Broker    │    │
│   ┌─────────┐                    │  • Rules Engine      │    │
│   │  ESP32  │───────────────────▶│  • Device Shadow     │    │
│   └─────────┘                    │  • Registry          │    │
│                                  └──────────┬───────────┘    │
│                                             │                │
│        ┌────────────────┬──────────────────┼────────────┐   │
│        │                │                  │            │   │
│        ▼                ▼                  ▼            ▼   │
│   ┌─────────┐    ┌──────────┐     ┌──────────┐  ┌────────┐ │
│   │   S3    │    │ DynamoDB │     │  Lambda  │  │Kinesis │ │
│   │ (저장)  │    │  (DB)    │     │ (처리)   │  │(스트림)│ │
│   └─────────┘    └──────────┘     └──────────┘  └────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 핵심 구성 요소

| 구성 요소 | 설명 |
|----------|------|
| **Device Gateway** | MQTT/HTTPS/WebSocket 연결 관리 |
| **Message Broker** | Pub/Sub 메시지 라우팅 |
| **Rules Engine** | 메시지 필터링 및 AWS 서비스 연동 |
| **Device Shadow** | 디바이스 상태의 가상 복제본 |
| **Registry** | 디바이스 ID 및 인증 관리 |

### 1.3 AWS CLI 설정

```bash
# AWS CLI 설치
pip install awscli

# 자격 증명 설정
aws configure
# AWS Access Key ID: your-access-key
# AWS Secret Access Key: your-secret-key
# Default region name: ap-northeast-2
# Default output format: json

# IoT 엔드포인트 확인
aws iot describe-endpoint --endpoint-type iot:Data-ATS
```

---

## 2. AWS IoT 디바이스 등록

### 2.1 인증서 생성

```bash
# 디바이스 디렉토리 생성
mkdir -p ~/iot-certs && cd ~/iot-certs

# 인증서 및 키 생성
aws iot create-keys-and-certificate \
    --set-as-active \
    --certificate-pem-outfile device.cert.pem \
    --public-key-outfile device.public.key \
    --private-key-outfile device.private.key

# 루트 CA 다운로드
wget https://www.amazontrust.com/repository/AmazonRootCA1.pem -O root-CA.crt
```

### 2.2 정책 생성 및 연결

```bash
# IoT 정책 생성
cat > iot-policy.json << 'EOF'
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "iot:Connect",
                "iot:Publish",
                "iot:Subscribe",
                "iot:Receive"
            ],
            "Resource": "*"
        }
    ]
}
EOF

aws iot create-policy \
    --policy-name IoTDevicePolicy \
    --policy-document file://iot-policy.json

# 인증서에 정책 연결
aws iot attach-policy \
    --policy-name IoTDevicePolicy \
    --target <인증서-ARN>

# Thing 생성
aws iot create-thing --thing-name RaspberryPi-001

# Thing에 인증서 연결
aws iot attach-thing-principal \
    --thing-name RaspberryPi-001 \
    --principal <인증서-ARN>
```

### 2.3 Python 연결 클라이언트

```python
#!/usr/bin/env python3
"""AWS IoT Core 연결"""

from awscrt import mqtt
from awsiot import mqtt_connection_builder
import json
import time

class AWSIoTClient:
    """AWS IoT Core 클라이언트"""

    def __init__(self, endpoint: str, cert_path: str, key_path: str,
                 ca_path: str, client_id: str):
        self.endpoint = endpoint
        self.client_id = client_id

        # MQTT 연결 생성
        self.connection = mqtt_connection_builder.mtls_from_path(
            endpoint=endpoint,
            port=8883,
            cert_filepath=cert_path,
            pri_key_filepath=key_path,
            ca_filepath=ca_path,
            client_id=client_id,
            clean_session=False,
            keep_alive_secs=30
        )

        self.connected = False

    def connect(self):
        """연결"""
        print(f"AWS IoT 연결 중: {self.endpoint}")

        connect_future = self.connection.connect()
        connect_future.result()

        self.connected = True
        print("연결 성공!")

    def disconnect(self):
        """연결 해제"""
        if self.connected:
            disconnect_future = self.connection.disconnect()
            disconnect_future.result()
            print("연결 해제됨")

    def publish(self, topic: str, payload: dict, qos: mqtt.QoS = mqtt.QoS.AT_LEAST_ONCE):
        """메시지 발행"""
        message = json.dumps(payload)

        publish_future, _ = self.connection.publish(
            topic=topic,
            payload=message,
            qos=qos
        )
        publish_future.result()

        print(f"발행: {topic} = {message}")

    def subscribe(self, topic: str, callback):
        """토픽 구독"""
        def on_message(topic, payload, **kwargs):
            message = json.loads(payload)
            callback(topic, message)

        subscribe_future, _ = self.connection.subscribe(
            topic=topic,
            qos=mqtt.QoS.AT_LEAST_ONCE,
            callback=on_message
        )
        subscribe_future.result()

        print(f"구독: {topic}")

# 사용 예
if __name__ == "__main__":
    # 인증서 경로
    ENDPOINT = "your-endpoint.iot.ap-northeast-2.amazonaws.com"
    CERT_PATH = "~/iot-certs/device.cert.pem"
    KEY_PATH = "~/iot-certs/device.private.key"
    CA_PATH = "~/iot-certs/root-CA.crt"

    client = AWSIoTClient(
        endpoint=ENDPOINT,
        cert_path=CERT_PATH,
        key_path=KEY_PATH,
        ca_path=CA_PATH,
        client_id="RaspberryPi-001"
    )

    try:
        client.connect()

        # 센서 데이터 발행
        for i in range(10):
            data = {
                "device_id": "RaspberryPi-001",
                "temperature": 25.5 + i * 0.1,
                "humidity": 60 + i,
                "timestamp": int(time.time())
            }
            client.publish("sensor/data", data)
            time.sleep(5)

    finally:
        client.disconnect()
```

---

## 3. GCP IoT (Pub/Sub)

> **중요 안내**: GCP IoT Core 서비스는 2023년 8월 16일에 종료되었습니다.
>
> Google은 IoT Core를 대체하여 **Cloud Pub/Sub + Cloud Functions** 조합을 권장합니다.
> 기존 IoT Core 사용자는 다음 대안을 고려하세요:
> - **Cloud Pub/Sub**: 메시지 브로커 (아래 예제)
> - **Clearblade IoT Core**: Google 파트너사의 IoT Core 호환 서비스
> - **AWS IoT Core**: AWS 대안
> - **Azure IoT Hub**: Microsoft 대안
>
> 이 레슨에서는 **Cloud Pub/Sub를 직접 사용**하는 방법을 다룹니다.

### 3.1 GCP Pub/Sub 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    GCP Pub/Sub 아키텍처                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Publisher                                 Subscriber       │
│   (IoT 디바이스)                            (처리 서비스)    │
│   ┌─────────┐        ┌──────────────┐      ┌─────────┐      │
│   │라즈베리 │──────▶│    Topic     │─────▶│ Cloud   │      │
│   │  파이   │        │  (sensor)    │      │ Function│      │
│   └─────────┘        └──────────────┘      └─────────┘      │
│                             │                               │
│                             ▼                               │
│                      ┌──────────────┐                       │
│                      │ Subscription │                       │
│                      │    (pull)    │                       │
│                      └──────────────┘                       │
│                             │                               │
│                             ▼                               │
│                      ┌──────────────┐                       │
│                      │   BigQuery   │                       │
│                      │   (분석)     │                       │
│                      └──────────────┘                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 GCP 설정

```bash
# gcloud CLI 설치
# https://cloud.google.com/sdk/docs/install

# 인증
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Pub/Sub API 활성화
gcloud services enable pubsub.googleapis.com

# 토픽 생성
gcloud pubsub topics create iot-sensor-data

# 구독 생성
gcloud pubsub subscriptions create iot-sensor-sub \
    --topic=iot-sensor-data

# 서비스 계정 생성
gcloud iam service-accounts create iot-device \
    --display-name="IoT Device Service Account"

# 권한 부여
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:iot-device@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/pubsub.publisher"

# 키 생성
gcloud iam service-accounts keys create ~/gcp-credentials.json \
    --iam-account=iot-device@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

### 3.3 Python Pub/Sub 클라이언트

```python
#!/usr/bin/env python3
"""GCP Pub/Sub 클라이언트"""

from google.cloud import pubsub_v1
import json
import time
import os

# 인증 설정
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gcp-credentials.json'

class GCPPubSubClient:
    """GCP Pub/Sub 클라이언트"""

    def __init__(self, project_id: str, topic_id: str):
        self.project_id = project_id
        self.topic_id = topic_id
        self.topic_path = f"projects/{project_id}/topics/{topic_id}"

        self.publisher = pubsub_v1.PublisherClient()

    def publish(self, data: dict, **attributes):
        """메시지 발행"""
        message = json.dumps(data).encode('utf-8')

        future = self.publisher.publish(
            self.topic_path,
            message,
            **attributes
        )

        message_id = future.result()
        print(f"발행됨: {message_id}")
        return message_id

    def publish_batch(self, messages: list):
        """배치 발행"""
        futures = []

        for data in messages:
            message = json.dumps(data).encode('utf-8')
            future = self.publisher.publish(self.topic_path, message)
            futures.append(future)

        # 모든 발행 완료 대기
        results = [f.result() for f in futures]
        print(f"배치 발행 완료: {len(results)}개")
        return results

class GCPPubSubSubscriber:
    """GCP Pub/Sub 구독자"""

    def __init__(self, project_id: str, subscription_id: str):
        self.project_id = project_id
        self.subscription_id = subscription_id
        self.subscription_path = f"projects/{project_id}/subscriptions/{subscription_id}"

        self.subscriber = pubsub_v1.SubscriberClient()

    def pull(self, max_messages: int = 10) -> list:
        """메시지 풀"""
        response = self.subscriber.pull(
            subscription=self.subscription_path,
            max_messages=max_messages
        )

        messages = []
        ack_ids = []

        for msg in response.received_messages:
            data = json.loads(msg.message.data.decode('utf-8'))
            messages.append({
                'data': data,
                'attributes': dict(msg.message.attributes),
                'message_id': msg.message.message_id
            })
            ack_ids.append(msg.ack_id)

        # ACK 전송
        if ack_ids:
            self.subscriber.acknowledge(
                subscription=self.subscription_path,
                ack_ids=ack_ids
            )

        return messages

    def subscribe(self, callback):
        """스트리밍 구독"""
        def on_message(message):
            data = json.loads(message.data.decode('utf-8'))
            callback(data, message.attributes)
            message.ack()

        streaming_pull = self.subscriber.subscribe(
            self.subscription_path,
            callback=on_message
        )

        print(f"구독 시작: {self.subscription_path}")
        return streaming_pull

# 사용 예
if __name__ == "__main__":
    PROJECT_ID = "your-project-id"

    # Publisher
    publisher = GCPPubSubClient(PROJECT_ID, "iot-sensor-data")

    for i in range(5):
        data = {
            "device_id": "pi-001",
            "temperature": 25.5 + i * 0.5,
            "timestamp": int(time.time())
        }
        publisher.publish(data, device_type="raspberry_pi")
        time.sleep(1)
```

---

## 4. MQTT 브릿지

### 4.1 로컬-클라우드 브릿지

```python
#!/usr/bin/env python3
"""MQTT 브릿지: 로컬 -> 클라우드"""

import paho.mqtt.client as mqtt
from google.cloud import pubsub_v1
import json
import threading
import time
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gcp-credentials.json'

class MQTTBridge:
    """로컬 MQTT를 클라우드로 브릿지"""

    def __init__(self, local_broker: str, gcp_project: str, gcp_topic: str):
        # 로컬 MQTT 클라이언트
        self.local_client = mqtt.Client(client_id="mqtt_bridge")
        self.local_client.on_connect = self._on_local_connect
        self.local_client.on_message = self._on_local_message

        self.local_broker = local_broker

        # GCP Pub/Sub
        self.publisher = pubsub_v1.PublisherClient()
        self.topic_path = f"projects/{gcp_project}/topics/{gcp_topic}"

        # 브릿지 설정
        self.topic_mapping = {
            "sensor/#": True,
            "device/+/status": True
        }

        self.stats = {
            "messages_received": 0,
            "messages_forwarded": 0,
            "errors": 0
        }

    def _on_local_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("로컬 브로커 연결됨")
            # 토픽 구독
            for topic in self.topic_mapping:
                client.subscribe(topic)
                print(f"구독: {topic}")

    def _on_local_message(self, client, userdata, msg):
        """로컬 메시지 수신 -> 클라우드로 전달"""
        self.stats["messages_received"] += 1

        try:
            # 메시지 파싱
            payload = msg.payload.decode('utf-8')

            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                data = {"raw": payload}

            # 메타데이터 추가
            cloud_message = {
                "source_topic": msg.topic,
                "data": data,
                "timestamp": time.time()
            }

            # 클라우드로 발행
            self._forward_to_cloud(cloud_message, msg.topic)
            self.stats["messages_forwarded"] += 1

        except Exception as e:
            print(f"브릿지 오류: {e}")
            self.stats["errors"] += 1

    def _forward_to_cloud(self, message: dict, source_topic: str):
        """클라우드로 메시지 전달"""
        data = json.dumps(message).encode('utf-8')

        future = self.publisher.publish(
            self.topic_path,
            data,
            source_topic=source_topic
        )
        future.result()  # 발행 완료 대기

    def start(self):
        """브릿지 시작"""
        print(f"MQTT 브릿지 시작: {self.local_broker} -> GCP")

        self.local_client.connect(self.local_broker, 1883)
        self.local_client.loop_start()

    def stop(self):
        """브릿지 중지"""
        self.local_client.loop_stop()
        self.local_client.disconnect()

        print(f"\n브릿지 통계:")
        print(f"  수신: {self.stats['messages_received']}")
        print(f"  전달: {self.stats['messages_forwarded']}")
        print(f"  오류: {self.stats['errors']}")

    def run(self):
        """메인 루프"""
        self.start()

        try:
            while True:
                time.sleep(10)
                print(f"브릿지 상태: 수신={self.stats['messages_received']}, "
                      f"전달={self.stats['messages_forwarded']}")
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

if __name__ == "__main__":
    bridge = MQTTBridge(
        local_broker="localhost",
        gcp_project="your-project-id",
        gcp_topic="iot-sensor-data"
    )
    bridge.run()
```

### 4.2 양방향 브릿지

```python
#!/usr/bin/env python3
"""양방향 MQTT-클라우드 브릿지"""

import paho.mqtt.client as mqtt
from google.cloud import pubsub_v1
import json
import threading
import time

class BidirectionalBridge:
    """양방향 브릿지"""

    def __init__(self, local_broker: str, gcp_project: str,
                 up_topic: str, down_topic: str, down_sub: str):
        # 로컬 MQTT
        self.local_client = mqtt.Client()
        self.local_broker = local_broker

        # GCP Pub/Sub
        self.gcp_project = gcp_project
        self.publisher = pubsub_v1.PublisherClient()
        self.subscriber = pubsub_v1.SubscriberClient()

        self.up_topic_path = f"projects/{gcp_project}/topics/{up_topic}"
        self.down_sub_path = f"projects/{gcp_project}/subscriptions/{down_sub}"

    def _on_local_message(self, client, userdata, msg):
        """로컬 -> 클라우드"""
        try:
            payload = json.loads(msg.payload.decode())
            data = json.dumps({
                "topic": msg.topic,
                "payload": payload
            }).encode()

            self.publisher.publish(self.up_topic_path, data)
            print(f"[UP] {msg.topic}")

        except Exception as e:
            print(f"업스트림 오류: {e}")

    def _on_cloud_message(self, message):
        """클라우드 -> 로컬"""
        try:
            data = json.loads(message.data.decode())
            topic = data.get("topic", "command")
            payload = data.get("payload", {})

            self.local_client.publish(topic, json.dumps(payload))
            print(f"[DOWN] {topic}")

            message.ack()

        except Exception as e:
            print(f"다운스트림 오류: {e}")
            message.nack()

    def start(self):
        """브릿지 시작"""
        # 로컬 MQTT 시작
        self.local_client.on_message = self._on_local_message
        self.local_client.connect(self.local_broker, 1883)
        self.local_client.subscribe("sensor/#")
        self.local_client.loop_start()

        # 클라우드 구독 시작
        self.streaming_pull = self.subscriber.subscribe(
            self.down_sub_path,
            callback=self._on_cloud_message
        )

        print("양방향 브릿지 시작")

    def stop(self):
        """브릿지 중지"""
        self.streaming_pull.cancel()
        self.local_client.loop_stop()
        self.local_client.disconnect()

    def run(self):
        self.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
```

---

## 5. 데이터 수집 및 분석

### 5.1 시계열 데이터 저장

```python
#!/usr/bin/env python3
"""IoT 데이터 저장 및 분석"""

from datetime import datetime, timedelta
import json
from typing import List, Dict
from dataclasses import dataclass

# InfluxDB 사용 (시계열 DB)
# pip install influxdb-client

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

@dataclass
class SensorData:
    device_id: str
    temperature: float
    humidity: float
    timestamp: datetime

class TimeSeriesDB:
    """시계열 데이터베이스 클라이언트"""

    def __init__(self, url: str, token: str, org: str, bucket: str):
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()
        self.bucket = bucket
        self.org = org

    def write_sensor_data(self, data: SensorData):
        """센서 데이터 저장"""
        point = Point("sensor_reading") \
            .tag("device_id", data.device_id) \
            .field("temperature", data.temperature) \
            .field("humidity", data.humidity) \
            .time(data.timestamp)

        self.write_api.write(bucket=self.bucket, record=point)

    def write_batch(self, data_list: List[SensorData]):
        """배치 저장"""
        points = []
        for data in data_list:
            point = Point("sensor_reading") \
                .tag("device_id", data.device_id) \
                .field("temperature", data.temperature) \
                .field("humidity", data.humidity) \
                .time(data.timestamp)
            points.append(point)

        self.write_api.write(bucket=self.bucket, record=points)

    def query_recent(self, device_id: str, hours: int = 24) -> List[Dict]:
        """최근 데이터 조회"""
        query = f'''
        from(bucket: "{self.bucket}")
            |> range(start: -{hours}h)
            |> filter(fn: (r) => r["device_id"] == "{device_id}")
            |> filter(fn: (r) => r["_measurement"] == "sensor_reading")
        '''

        tables = self.query_api.query(query, org=self.org)

        results = []
        for table in tables:
            for record in table.records:
                results.append({
                    "time": record.get_time(),
                    "field": record.get_field(),
                    "value": record.get_value()
                })

        return results

    def get_statistics(self, device_id: str, hours: int = 24) -> Dict:
        """통계 조회"""
        query = f'''
        from(bucket: "{self.bucket}")
            |> range(start: -{hours}h)
            |> filter(fn: (r) => r["device_id"] == "{device_id}")
            |> filter(fn: (r) => r["_field"] == "temperature")
            |> mean()
        '''

        tables = self.query_api.query(query, org=self.org)

        stats = {}
        for table in tables:
            for record in table.records:
                stats["mean_temperature"] = record.get_value()

        return stats

    def close(self):
        self.client.close()
```

### 5.2 대시보드 데이터 API

```python
#!/usr/bin/env python3
"""IoT 대시보드 API"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta
import json

app = Flask(__name__)
CORS(app)

# 데이터 스토어 (실제로는 DB 연결)
data_store = {
    "devices": {},
    "readings": []
}

@app.route('/api/devices')
def list_devices():
    """디바이스 목록"""
    return jsonify({
        "devices": list(data_store["devices"].values()),
        "count": len(data_store["devices"])
    })

@app.route('/api/devices/<device_id>/readings')
def get_readings(device_id):
    """디바이스 데이터 조회"""
    hours = request.args.get('hours', 24, type=int)
    limit = request.args.get('limit', 100, type=int)

    cutoff = datetime.now() - timedelta(hours=hours)

    readings = [
        r for r in data_store["readings"]
        if r["device_id"] == device_id and
           datetime.fromisoformat(r["timestamp"]) > cutoff
    ]

    readings = sorted(readings, key=lambda x: x["timestamp"], reverse=True)[:limit]

    return jsonify({
        "device_id": device_id,
        "readings": readings,
        "count": len(readings)
    })

@app.route('/api/devices/<device_id>/stats')
def get_stats(device_id):
    """디바이스 통계"""
    hours = request.args.get('hours', 24, type=int)
    cutoff = datetime.now() - timedelta(hours=hours)

    readings = [
        r for r in data_store["readings"]
        if r["device_id"] == device_id and
           datetime.fromisoformat(r["timestamp"]) > cutoff
    ]

    if not readings:
        return jsonify({"error": "No data"}), 404

    temps = [r["temperature"] for r in readings if "temperature" in r]
    humids = [r["humidity"] for r in readings if "humidity" in r]

    stats = {
        "device_id": device_id,
        "period_hours": hours,
        "reading_count": len(readings)
    }

    if temps:
        stats["temperature"] = {
            "min": min(temps),
            "max": max(temps),
            "avg": sum(temps) / len(temps)
        }

    if humids:
        stats["humidity"] = {
            "min": min(humids),
            "max": max(humids),
            "avg": sum(humids) / len(humids)
        }

    return jsonify(stats)

@app.route('/api/alerts')
def get_alerts():
    """알림 조회"""
    alerts = []

    # 임계값 체크
    for device_id, device in data_store["devices"].items():
        last_reading = device.get("last_reading")
        if last_reading:
            if last_reading.get("temperature", 0) > 35:
                alerts.append({
                    "device_id": device_id,
                    "type": "high_temperature",
                    "value": last_reading["temperature"],
                    "threshold": 35
                })

    return jsonify({"alerts": alerts, "count": len(alerts)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
```

---

## 6. 종합 IoT 파이프라인

```python
#!/usr/bin/env python3
"""종합 IoT 데이터 파이프라인"""

import paho.mqtt.client as mqtt
from google.cloud import pubsub_v1
import json
import time
import threading
from datetime import datetime
from dataclasses import dataclass
from typing import Callable, List

@dataclass
class IoTMessage:
    device_id: str
    data: dict
    timestamp: datetime
    source: str

class IoTPipeline:
    """IoT 데이터 파이프라인"""

    def __init__(self, config: dict):
        self.config = config
        self.handlers: List[Callable] = []

        # 로컬 MQTT
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_message = self._on_mqtt_message

        # 클라우드 (옵션)
        if config.get("gcp_enabled"):
            self.publisher = pubsub_v1.PublisherClient()

    def add_handler(self, handler: Callable):
        """메시지 핸들러 추가"""
        self.handlers.append(handler)

    def _on_mqtt_message(self, client, userdata, msg):
        """MQTT 메시지 처리"""
        try:
            payload = json.loads(msg.payload.decode())

            message = IoTMessage(
                device_id=payload.get("device_id", "unknown"),
                data=payload,
                timestamp=datetime.now(),
                source=msg.topic
            )

            # 핸들러 실행
            for handler in self.handlers:
                try:
                    handler(message)
                except Exception as e:
                    print(f"핸들러 오류: {e}")

        except Exception as e:
            print(f"메시지 처리 오류: {e}")

    def start(self):
        """파이프라인 시작"""
        broker = self.config.get("mqtt_broker", "localhost")
        topics = self.config.get("mqtt_topics", ["sensor/#"])

        self.mqtt_client.connect(broker, 1883)

        for topic in topics:
            self.mqtt_client.subscribe(topic)
            print(f"구독: {topic}")

        self.mqtt_client.loop_start()
        print("IoT 파이프라인 시작됨")

    def stop(self):
        """파이프라인 중지"""
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()

# 핸들러 예시
def log_handler(message: IoTMessage):
    """로깅 핸들러"""
    print(f"[{message.timestamp}] {message.device_id}: {message.data}")

def alert_handler(message: IoTMessage):
    """알림 핸들러"""
    temp = message.data.get("temperature")
    if temp and temp > 35:
        print(f"[ALERT] 고온 감지: {message.device_id} = {temp}°C")

def cloud_handler(message: IoTMessage, publisher, topic_path):
    """클라우드 전송 핸들러"""
    data = json.dumps({
        "device_id": message.device_id,
        "data": message.data,
        "timestamp": message.timestamp.isoformat()
    }).encode()

    publisher.publish(topic_path, data)

# 사용 예
if __name__ == "__main__":
    config = {
        "mqtt_broker": "localhost",
        "mqtt_topics": ["sensor/#", "device/+/status"],
        "gcp_enabled": False
    }

    pipeline = IoTPipeline(config)
    pipeline.add_handler(log_handler)
    pipeline.add_handler(alert_handler)

    pipeline.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pipeline.stop()
```

---

## 연습 문제

### 문제 1: AWS IoT 연동
1. AWS IoT Core에 디바이스를 등록하세요.
2. 센서 데이터를 AWS IoT로 발행하세요.

### 문제 2: 데이터 분석
1. 수집된 센서 데이터의 일일 통계를 계산하세요.
2. 이상치 감지 알림을 구현하세요.

### 문제 3: 대시보드
1. 클라우드 데이터를 시각화하는 대시보드를 만드세요.
2. 실시간 데이터 업데이트를 구현하세요.

---

## 마무리

이 레슨에서는 IoT 디바이스를 클라우드 서비스와 연동하는 방법을 학습했습니다. AWS IoT Core와 GCP Pub/Sub을 활용하여 대규모 IoT 시스템을 구축할 수 있습니다.

### 전체 학습 요약

| 레슨 | 주제 | 핵심 기술 |
|------|------|----------|
| 01 | IoT 개요 | 아키텍처, 프로토콜 |
| 02 | 라즈베리파이 | 설정, SSH, GPIO |
| 03 | GPIO 제어 | gpiozero, 센서 |
| 04 | WiFi | 소켓, HTTP |
| 05 | BLE | bleak, GATT |
| 06 | MQTT | paho-mqtt, Mosquitto |
| 07 | REST API | Flask, JSON |
| 08-09 | Edge AI | TFLite, ONNX |
| 10 | 홈 자동화 | 릴레이, 웹 대시보드 |
| 11 | 영상 분석 | picamera2, 모션 감지 |
| 12 | 클라우드 IoT | AWS, GCP, 브릿지 |

---

*최종 업데이트: 2026-02-01*
