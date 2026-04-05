"""
Django Advanced — Channels (WebSocket) and Celery (Background Tasks)
Demonstrates: WebSocket consumers, Celery tasks, Redis pub/sub.

Setup:
    pip install channels channels-redis celery[redis] django-celery-results
"""

# === Channels: WebSocket Consumer ===

"""
# consumers.py
import json
from channels.generic.websocket import AsyncJsonWebsocketConsumer


class ChatConsumer(AsyncJsonWebsocketConsumer):
    \"\"\"WebSocket consumer for real-time chat.\"\"\"

    async def connect(self):
        self.room_name = self.scope["url_route"]["kwargs"]["room_name"]
        self.room_group = f"chat_{self.room_name}"

        # Join room group
        await self.channel_layer.group_add(
            self.room_group,
            self.channel_name,
        )
        await self.accept()

        await self.send_json({
            "type": "system",
            "message": f"Connected to room: {self.room_name}",
        })

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.room_group,
            self.channel_name,
        )

    async def receive_json(self, content):
        message = content.get("message", "")
        username = content.get("username", "Anonymous")

        # Broadcast to room group
        await self.channel_layer.group_send(
            self.room_group,
            {
                "type": "chat.message",
                "message": message,
                "username": username,
            },
        )

    async def chat_message(self, event):
        \"\"\"Handle chat.message events from the group.\"\"\"
        await self.send_json({
            "type": "message",
            "message": event["message"],
            "username": event["username"],
        })
"""


# === Channels: Routing ===

"""
# routing.py
from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r"ws/chat/(?P<room_name>\w+)/$", consumers.ChatConsumer.as_asgi()),
]

# asgi.py
import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myproject.settings")

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            # import your routing.websocket_urlpatterns here
        )
    ),
})
"""


# === Celery Configuration ===

"""
# celery.py (in project root, same level as settings.py)
import os
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myproject.settings")

app = Celery("myproject")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()

# settings.py
CELERY_BROKER_URL = "redis://localhost:6379/0"
CELERY_RESULT_BACKEND = "django-db"
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_TIMEZONE = "UTC"
"""


# === Celery Tasks ===

"""
# tasks.py
from celery import shared_task
from django.core.mail import send_mail
from django.utils import timezone
import logging

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3)
def send_welcome_email(self, user_id):
    \"\"\"Send welcome email with automatic retry on failure.\"\"\"
    try:
        from django.contrib.auth.models import User
        user = User.objects.get(id=user_id)
        send_mail(
            subject="Welcome!",
            message=f"Hello {user.username}, welcome to our platform!",
            from_email="noreply@example.com",
            recipient_list=[user.email],
        )
        logger.info(f"Welcome email sent to {user.email}")
    except Exception as exc:
        logger.error(f"Email failed: {exc}")
        raise self.retry(exc=exc, countdown=60 * (self.request.retries + 1))


@shared_task
def generate_report(report_type, start_date, end_date):
    \"\"\"Generate a report asynchronously.\"\"\"
    import time
    time.sleep(5)  # Simulate heavy computation
    return {
        "type": report_type,
        "start": start_date,
        "end": end_date,
        "generated_at": timezone.now().isoformat(),
        "data": {"total_users": 1000, "active_users": 750},
    }


@shared_task
def cleanup_old_sessions():
    \"\"\"Periodic task to clean up expired sessions.\"\"\"
    from django.contrib.sessions.models import Session
    expired = Session.objects.filter(expire_date__lt=timezone.now())
    count = expired.count()
    expired.delete()
    return f"Deleted {count} expired sessions"
"""


# === Using Celery Tasks in Views ===

"""
# views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from .tasks import generate_report, send_welcome_email


class ReportView(APIView):
    def post(self, request):
        task = generate_report.delay(
            report_type=request.data.get("type", "summary"),
            start_date=request.data.get("start"),
            end_date=request.data.get("end"),
        )
        return Response({
            "task_id": task.id,
            "status": "Processing",
        })

    def get(self, request, task_id):
        from celery.result import AsyncResult
        result = AsyncResult(task_id)
        return Response({
            "task_id": task_id,
            "status": result.status,
            "result": result.result if result.ready() else None,
        })
"""


# === Celery Beat (Periodic Tasks) ===

"""
# settings.py
from celery.schedules import crontab

CELERY_BEAT_SCHEDULE = {
    "cleanup-sessions-daily": {
        "task": "blog.tasks.cleanup_old_sessions",
        "schedule": crontab(hour=3, minute=0),  # Run at 3:00 AM
    },
    "generate-daily-report": {
        "task": "blog.tasks.generate_report",
        "schedule": crontab(hour=6, minute=0),
        "args": ("daily", None, None),
    },
}

# Run beat: celery -A myproject beat -l info
# Run worker: celery -A myproject worker -l info
"""


# --- Standalone Demo ---

if __name__ == "__main__":
    print("Django Advanced: Channels + Celery Demo")
    print("=" * 50)
    print()
    print("Channels (WebSocket):")
    print("  - AsyncJsonWebsocketConsumer for real-time chat")
    print("  - Room-based group messaging via channel_layer")
    print("  - AuthMiddlewareStack for WebSocket authentication")
    print()
    print("Celery (Background Tasks):")
    print("  - @shared_task with auto-retry (max_retries, countdown)")
    print("  - AsyncResult for task status polling")
    print("  - Celery Beat for periodic/scheduled tasks")
    print("  - Redis as broker, django-db as result backend")
