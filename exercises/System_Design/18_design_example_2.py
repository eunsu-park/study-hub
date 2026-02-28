"""
Exercises for Lesson 18: Practical Design Examples 2
Topic: System_Design

Solutions to practice problems from the lesson.
Covers news feed extensions, chat system encryption, and notification batching.
"""

import time
import random
import hashlib
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# === Exercise 1: News Feed Extension ===
# Problem: Design ad insertion, trending recommendations, "not interested" feedback.

@dataclass
class FeedItem:
    item_id: str
    author_id: str
    content: str
    created_at: float
    item_type: str = "post"  # "post", "ad", "trending"
    score: float = 0.0
    hidden: bool = False


class NewsFeedService:
    """News feed with ad insertion, trending, and feedback handling."""

    def __init__(self):
        self.posts = {}  # post_id -> FeedItem
        self.follows = defaultdict(set)  # user_id -> set of followed user_ids
        self.hidden_topics = defaultdict(set)  # user_id -> set of hidden topics/authors
        self.engagement = defaultdict(lambda: {"views": 0, "likes": 0, "shares": 0})
        self.ads = []

    def add_post(self, author_id, content, topics=None):
        post_id = f"post_{len(self.posts)}"
        item = FeedItem(
            item_id=post_id,
            author_id=author_id,
            content=content,
            created_at=time.time(),
        )
        self.posts[post_id] = item
        return post_id

    def add_ad(self, content, target_segment="all"):
        ad = FeedItem(
            item_id=f"ad_{len(self.ads)}",
            author_id="sponsor",
            content=content,
            created_at=time.time(),
            item_type="ad",
        )
        self.ads.append(ad)

    def get_feed(self, user_id, limit=20):
        """Generate personalized feed with ads and trending."""
        # Collect posts from followed users
        followed = self.follows.get(user_id, set())
        candidate_posts = [
            p for p in self.posts.values()
            if p.author_id in followed
            and p.author_id not in self.hidden_topics[user_id]
            and not p.hidden
        ]

        # Sort by recency + engagement score
        for post in candidate_posts:
            eng = self.engagement[post.item_id]
            recency = max(0, 1 - (time.time() - post.created_at) / 86400)
            engagement_score = eng["likes"] * 2 + eng["shares"] * 5
            post.score = recency * 100 + engagement_score

        candidate_posts.sort(key=lambda x: x.score, reverse=True)

        # Insert ads every 5th position
        feed = []
        ad_idx = 0
        for i, post in enumerate(candidate_posts[:limit]):
            feed.append(post)
            if (i + 1) % 5 == 0 and ad_idx < len(self.ads):
                feed.append(self.ads[ad_idx])
                ad_idx += 1

        # Add trending posts (from non-followed users)
        trending = self._get_trending(user_id, count=3)
        for t in trending:
            t.item_type = "trending"
        feed.extend(trending)

        return feed[:limit]

    def _get_trending(self, user_id, count=3):
        """Get trending posts the user hasn't seen."""
        followed = self.follows.get(user_id, set())
        trending = [
            p for p in self.posts.values()
            if p.author_id not in followed
            and self.engagement[p.item_id]["likes"] > 5
        ]
        trending.sort(key=lambda x: self.engagement[x.item_id]["likes"], reverse=True)
        return trending[:count]

    def mark_not_interested(self, user_id, item_id, reason="not_interested"):
        """Handle 'not interested' feedback."""
        post = self.posts.get(item_id)
        if post:
            self.hidden_topics[user_id].add(post.author_id)
            post.hidden = True  # For this user's future feeds


def exercise_1():
    """News feed with ads, trending, and feedback."""
    print("News Feed Extensions:")
    print("=" * 60)

    feed_svc = NewsFeedService()

    # Set up users and follows
    feed_svc.follows["viewer"] = {"alice", "bob", "charlie"}

    # Create posts
    for author in ["alice", "bob", "charlie", "trending_user"]:
        for i in range(5):
            pid = feed_svc.add_post(author, f"{author}'s post #{i}")
            # Simulate engagement
            feed_svc.engagement[pid]["likes"] = random.randint(0, 20)
            feed_svc.engagement[pid]["shares"] = random.randint(0, 5)

    # Add ads
    feed_svc.add_ad("Buy our product! 50% off")
    feed_svc.add_ad("New course: System Design")

    # Get feed
    print("\n  Feed for 'viewer':")
    feed = feed_svc.get_feed("viewer", limit=15)
    for i, item in enumerate(feed):
        type_tag = f"[{item.item_type.upper():>8}]"
        print(f"    {i+1:2d}. {type_tag} {item.content[:40]} "
              f"(score={item.score:.0f})")

    # Mark not interested
    print("\n  User marks 'not interested' on alice's post")
    if feed:
        feed_svc.mark_not_interested("viewer", feed[0].item_id)
        print(f"  Alice's posts now hidden from viewer's feed")

    new_feed = feed_svc.get_feed("viewer", limit=15)
    alice_posts = [f for f in new_feed if f.author_id == "alice"]
    print(f"  Alice's posts in new feed: {len(alice_posts)}")


# === Exercise 2: Chat System Extension ===
# Problem: E2E encryption, message edit/delete within 24h, signaling.

class E2EChatSystem:
    """Chat system with end-to-end encryption simulation."""

    def __init__(self):
        self.messages = {}  # msg_id -> encrypted message
        self.user_keys = {}  # user_id -> (public_key, private_key) simulated

    def generate_keys(self, user_id):
        """Simulate key pair generation (Diffie-Hellman style)."""
        # In reality, use asymmetric encryption (RSA, X25519)
        private_key = random.randint(1000, 9999)
        public_key = private_key * 7 % 10000  # Simplified
        self.user_keys[user_id] = (public_key, private_key)
        return public_key

    def _derive_shared_secret(self, user_a, user_b):
        """Derive shared secret from two users' keys."""
        # Simplified: XOR of public keys
        pub_a = self.user_keys[user_a][0]
        pub_b = self.user_keys[user_b][0]
        return pub_a ^ pub_b

    def encrypt_message(self, sender, receiver, plaintext):
        """Encrypt message with shared secret."""
        secret = self._derive_shared_secret(sender, receiver)
        # Simplified XOR encryption
        key = str(secret).encode()
        encrypted = bytearray()
        for i, byte in enumerate(plaintext.encode()):
            encrypted.append(byte ^ key[i % len(key)])
        return encrypted.hex()

    def decrypt_message(self, sender, receiver, encrypted_hex):
        """Decrypt message with shared secret."""
        secret = self._derive_shared_secret(sender, receiver)
        encrypted = bytes.fromhex(encrypted_hex)
        key = str(secret).encode()
        decrypted = bytearray()
        for i, byte in enumerate(encrypted):
            decrypted.append(byte ^ key[i % len(key)])
        return decrypted.decode()

    def send_message(self, sender, receiver, plaintext):
        """Send an E2E encrypted message."""
        msg_id = f"msg_{len(self.messages)}"
        encrypted = self.encrypt_message(sender, receiver, plaintext)
        self.messages[msg_id] = {
            "sender": sender,
            "receiver": receiver,
            "encrypted_content": encrypted,
            "created_at": time.time(),
            "edited": False,
            "deleted": False,
        }
        return msg_id

    def edit_message(self, msg_id, sender, new_plaintext):
        """Edit message within 24 hours."""
        msg = self.messages.get(msg_id)
        if not msg:
            return False, "Message not found"
        if msg["sender"] != sender:
            return False, "Not the sender"
        if time.time() - msg["created_at"] > 86400:
            return False, "Edit window expired (24h)"

        encrypted = self.encrypt_message(sender, msg["receiver"], new_plaintext)
        msg["encrypted_content"] = encrypted
        msg["edited"] = True
        return True, "Message edited"

    def delete_message(self, msg_id, sender):
        """Delete message within 24 hours."""
        msg = self.messages.get(msg_id)
        if not msg:
            return False, "Message not found"
        if msg["sender"] != sender:
            return False, "Not the sender"
        if time.time() - msg["created_at"] > 86400:
            return False, "Delete window expired (24h)"

        msg["deleted"] = True
        msg["encrypted_content"] = ""
        return True, "Message deleted"


def exercise_2():
    """Chat system with E2E encryption and message management."""
    print("Chat System: E2E Encryption + Edit/Delete:")
    print("=" * 60)

    chat = E2EChatSystem()

    # Key exchange
    print("\n--- Key Exchange ---")
    alice_pub = chat.generate_keys("alice")
    bob_pub = chat.generate_keys("bob")
    print(f"  Alice's public key: {alice_pub}")
    print(f"  Bob's public key: {bob_pub}")

    # Send encrypted message
    print("\n--- E2E Encrypted Messaging ---")
    msg_id = chat.send_message("alice", "bob", "Hello Bob! Secret meeting at 3pm.")
    msg = chat.messages[msg_id]
    print(f"  Alice sends: 'Hello Bob! Secret meeting at 3pm.'")
    print(f"  Stored (encrypted): {msg['encrypted_content'][:40]}...")

    # Bob decrypts
    decrypted = chat.decrypt_message("alice", "bob", msg["encrypted_content"])
    print(f"  Bob decrypts: '{decrypted}'")

    # Server cannot read
    print(f"  Server sees: encrypted hex data (cannot read)")

    # Edit message
    print("\n--- Message Edit ---")
    ok, status = chat.edit_message(msg_id, "alice", "Meeting moved to 4pm.")
    print(f"  Alice edits: {status}")
    decrypted = chat.decrypt_message("alice", "bob",
                                      chat.messages[msg_id]["encrypted_content"])
    print(f"  Bob reads edited: '{decrypted}'")

    # Delete message
    print("\n--- Message Delete ---")
    ok, status = chat.delete_message(msg_id, "alice")
    print(f"  Alice deletes: {status}")
    print(f"  Message content: '{chat.messages[msg_id]['encrypted_content']}'")

    # Video call signaling
    print("\n--- Video/Voice Call Signaling ---")
    print("  WebRTC signaling flow:")
    print("    1. Alice sends 'offer' SDP through chat server")
    print("    2. Bob receives offer, sends 'answer' SDP")
    print("    3. ICE candidates exchanged (STUN/TURN)")
    print("    4. Direct peer-to-peer connection established")
    print("    5. Media streams encrypted end-to-end (SRTP)")


# === Exercise 3: Notification System Optimization ===
# Problem: Multi-language, batching, A/B testing for notifications.

class NotificationSystem:
    """Notification system with batching, i18n, and A/B testing."""

    def __init__(self):
        self.templates = {}
        self.notifications_sent = []
        self.batch_queue = defaultdict(list)
        self.batch_interval = 300  # 5 minutes

    def register_template(self, template_id, translations):
        """Register notification template with translations."""
        self.templates[template_id] = translations

    def render(self, template_id, language, params):
        """Render notification in specified language."""
        templates = self.templates.get(template_id, {})
        template = templates.get(language, templates.get("en", ""))
        for key, value in params.items():
            template = template.replace(f"{{{key}}}", str(value))
        return template

    def send(self, user_id, template_id, language, params, priority="normal"):
        """Send or batch a notification."""
        rendered = self.render(template_id, language, params)

        if priority == "high":
            # Send immediately
            self._deliver(user_id, rendered, template_id)
        else:
            # Add to batch queue
            self.batch_queue[user_id].append({
                "template_id": template_id,
                "rendered": rendered,
                "timestamp": time.time(),
            })

    def flush_batches(self):
        """Flush batched notifications (grouped)."""
        for user_id, notifications in self.batch_queue.items():
            if len(notifications) == 1:
                self._deliver(user_id, notifications[0]["rendered"],
                              notifications[0]["template_id"])
            else:
                # Group similar notifications
                groups = defaultdict(list)
                for n in notifications:
                    groups[n["template_id"]].append(n)

                for template_id, group in groups.items():
                    if len(group) > 1:
                        summary = f"{len(group)} new notifications: " + \
                                  ", ".join(n["rendered"][:30] for n in group[:3])
                        if len(group) > 3:
                            summary += f" and {len(group)-3} more"
                        self._deliver(user_id, summary, template_id)
                    else:
                        self._deliver(user_id, group[0]["rendered"], template_id)

        self.batch_queue.clear()

    def _deliver(self, user_id, content, template_id):
        self.notifications_sent.append({
            "user_id": user_id,
            "content": content,
            "template_id": template_id,
            "delivered_at": time.time(),
        })

    def ab_test(self, template_id, variants, user_id):
        """A/B test notification copy."""
        # Consistent assignment based on user_id hash
        variant_idx = hash(user_id) % len(variants)
        return variants[variant_idx]


def exercise_3():
    """Notification system optimization."""
    print("Notification System Optimization:")
    print("=" * 60)

    notif = NotificationSystem()

    # 1. Multi-language templates
    print("\n--- Multi-Language Notifications ---")
    notif.register_template("order_shipped", {
        "en": "Your order #{order_id} has been shipped! Track it here.",
        "ko": "주문 #{order_id}이(가) 발송되었습니다! 여기서 추적하세요.",
        "ja": "注文 #{order_id} が発送されました！ここで追跡できます。",
        "es": "Tu pedido #{order_id} ha sido enviado. Rastralo aqui.",
    })

    for lang in ["en", "ko", "ja", "es"]:
        rendered = notif.render("order_shipped", lang, {"order_id": "12345"})
        print(f"  [{lang}] {rendered}")

    # 2. Notification batching
    print("\n--- Notification Batching ---")
    notif.register_template("new_like", {
        "en": "{user} liked your post",
        "ko": "{user}님이 회원님의 게시물을 좋아합니다",
    })

    # Multiple likes come in rapid succession
    for user in ["Alice", "Bob", "Charlie", "Dave", "Eve",
                  "Frank", "Grace", "Henry"]:
        notif.send("recipient_1", "new_like", "en",
                   {"user": user}, priority="normal")

    print(f"  Queued: {len(notif.batch_queue['recipient_1'])} notifications")

    notif.flush_batches()
    print(f"  After batching: {len(notif.notifications_sent)} notification(s) sent")
    for n in notif.notifications_sent:
        print(f"    -> {n['content'][:70]}...")

    # 3. A/B testing
    print("\n--- A/B Testing Notification Copy ---")
    variants = [
        "Your order is on its way! Track it now.",
        "Great news! Your package just shipped. Tap to track.",
        "Shipping update: Your order left the warehouse.",
    ]

    variant_counts = defaultdict(int)
    for i in range(300):
        user_id = f"user_{i}"
        variant = notif.ab_test("order_shipped", variants, user_id)
        variant_counts[variant] += 1

    print(f"  A/B test results (300 users, 3 variants):")
    for variant, count in variant_counts.items():
        print(f"    '{variant[:50]}...': {count} users ({count/300:.1%})")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: News Feed Extensions ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Chat System Security ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Notification Optimization ===")
    print("=" * 60)
    exercise_3()

    print("\nAll exercises completed!")
