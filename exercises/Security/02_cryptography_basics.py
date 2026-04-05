"""
Exercise Solutions: Cryptography Basics
=======================================
Lesson 02 from Security topic.

Demonstrates secure cryptographic patterns using the Python `cryptography`
library. NEVER includes actual exploit code.

Dependencies:
    pip install cryptography
"""

import base64
import os
import time
from dataclasses import dataclass

from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding, ed25519, x25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305


# ---------------------------------------------------------------------------
# Exercise 1: Symmetric Encryption (Password-based AES-GCM)
# ---------------------------------------------------------------------------

def derive_key_from_password(password: str, salt: bytes) -> bytes:
    """Derive a 256-bit AES key from a password using PBKDF2."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=600_000,
    )
    return kdf.derive(password.encode())


def encrypt_with_password(plaintext: str, password: str) -> str:
    """
    Encrypt plaintext with AES-256-GCM using a password-derived key.
    Returns base64(salt + nonce + ciphertext).
    """
    salt = os.urandom(16)
    key = derive_key_from_password(password, salt)
    nonce = os.urandom(12)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext.encode(), None)
    # Pack: salt(16) + nonce(12) + ciphertext(variable)
    return base64.b64encode(salt + nonce + ciphertext).decode()


def decrypt_with_password(encoded: str, password: str) -> str:
    """
    Decrypt a base64-encoded message produced by encrypt_with_password.
    """
    raw = base64.b64decode(encoded)
    salt = raw[:16]
    nonce = raw[16:28]
    ciphertext = raw[28:]
    key = derive_key_from_password(password, salt)
    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    return plaintext.decode()


def exercise_1_symmetric_encryption():
    """Demonstrate password-based AES-GCM encryption and decryption."""
    message = "Sensitive data that must remain confidential"
    password = "strong-passphrase-example-2025"

    encrypted = encrypt_with_password(message, password)
    print(f"Original:  {message}")
    print(f"Encrypted: {encrypted[:60]}...")

    decrypted = decrypt_with_password(encrypted, password)
    print(f"Decrypted: {decrypted}")
    assert decrypted == message, "Decryption failed!"
    print("Symmetric encryption/decryption verified successfully.")


# ---------------------------------------------------------------------------
# Exercise 2: Hybrid Encryption (RSA + AES-GCM, PGP-like)
# ---------------------------------------------------------------------------

def generate_rsa_keypair():
    """Generate an RSA-4096 key pair."""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=4096,
    )
    return private_key, private_key.public_key()


def hybrid_encrypt(message: bytes, sender_private_key, receiver_public_key):
    """
    PGP-like hybrid encryption:
    1. Sign message with sender's private key
    2. Generate random AES key
    3. Encrypt message with AES-GCM
    4. Encrypt AES key with receiver's RSA public key
    """
    # Step 1: Sign the message
    from cryptography.hazmat.primitives.asymmetric import utils
    signature = sender_private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )

    # Step 2: Generate random AES-256 key
    aes_key = AESGCM.generate_key(bit_length=256)

    # Step 3: Encrypt message with AES-GCM
    nonce = os.urandom(12)
    aesgcm = AESGCM(aes_key)
    ciphertext = aesgcm.encrypt(nonce, message, None)

    # Step 4: Encrypt AES key with receiver's RSA public key
    encrypted_key = receiver_public_key.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )

    return {
        "encrypted_key": encrypted_key,
        "nonce": nonce,
        "ciphertext": ciphertext,
        "signature": signature,
    }


def hybrid_decrypt(package: dict, receiver_private_key, sender_public_key) -> bytes:
    """
    Decrypt and verify a hybrid-encrypted package.
    """
    # Step 1: Decrypt AES key with receiver's private RSA key
    aes_key = receiver_private_key.decrypt(
        package["encrypted_key"],
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )

    # Step 2: Decrypt message with AES-GCM
    aesgcm = AESGCM(aes_key)
    plaintext = aesgcm.decrypt(package["nonce"], package["ciphertext"], None)

    # Step 3: Verify signature
    sender_public_key.verify(
        package["signature"],
        plaintext,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )

    return plaintext


def exercise_2_hybrid_encryption():
    """Demonstrate PGP-like hybrid encryption with RSA + AES-GCM."""
    alice_priv, alice_pub = generate_rsa_keypair()
    bob_priv, bob_pub = generate_rsa_keypair()

    message = b"Hello Bob, this is a signed and encrypted message from Alice."

    # Alice encrypts and signs for Bob
    package = hybrid_encrypt(message, alice_priv, bob_pub)
    print(f"Encrypted key size: {len(package['encrypted_key'])} bytes")
    print(f"Ciphertext size:    {len(package['ciphertext'])} bytes")

    # Bob decrypts and verifies
    decrypted = hybrid_decrypt(package, bob_priv, alice_pub)
    print(f"Decrypted: {decrypted.decode()}")
    assert decrypted == message
    print("Hybrid encryption with signature verified successfully.")


# ---------------------------------------------------------------------------
# Exercise 3: Key Exchange Protocol (X25519 + ChaCha20-Poly1305)
# ---------------------------------------------------------------------------

@dataclass
class SecureChatParticipant:
    """A participant in a secure chat using X25519 key exchange."""
    name: str
    _private_key: x25519.X25519PrivateKey = None
    _send_key: bytes = None
    _recv_key: bytes = None
    _send_counter: int = 0
    _recv_counter: int = 0
    _seen_nonces: set = None

    def __post_init__(self):
        self._private_key = x25519.X25519PrivateKey.generate()
        self._seen_nonces = set()

    @property
    def public_key(self) -> x25519.X25519PublicKey:
        return self._private_key.public_key()

    def derive_keys(self, peer_public_key: x25519.X25519PublicKey, is_initiator: bool):
        """Derive separate send/receive keys using HKDF."""
        shared_secret = self._private_key.exchange(peer_public_key)

        # Derive two keys from the shared secret
        key_material = HKDF(
            algorithm=hashes.SHA256(),
            length=64,  # 32 bytes for each direction
            salt=None,
            info=b"secure-chat-v1",
        ).derive(shared_secret)

        key_a = key_material[:32]  # Initiator -> Responder
        key_b = key_material[32:]  # Responder -> Initiator

        if is_initiator:
            self._send_key = key_a
            self._recv_key = key_b
        else:
            self._send_key = key_b
            self._recv_key = key_a

    def encrypt_message(self, plaintext: str) -> dict:
        """Encrypt a message with ChaCha20-Poly1305 and counter-based nonce."""
        self._send_counter += 1
        # Build nonce from counter (12 bytes for ChaCha20-Poly1305)
        nonce = self._send_counter.to_bytes(12, "big")
        chacha = ChaCha20Poly1305(self._send_key)
        ciphertext = chacha.encrypt(nonce, plaintext.encode(), None)
        return {
            "ciphertext": ciphertext,
            "counter": self._send_counter,
        }

    def decrypt_message(self, package: dict) -> str:
        """Decrypt a message, checking for replay attacks."""
        counter = package["counter"]

        # Replay attack detection
        if counter in self._seen_nonces:
            raise ValueError("Replay attack detected: duplicate counter")
        if counter <= self._recv_counter:
            raise ValueError("Replay attack detected: counter too old")

        self._seen_nonces.add(counter)
        self._recv_counter = counter

        nonce = counter.to_bytes(12, "big")
        chacha = ChaCha20Poly1305(self._recv_key)
        plaintext = chacha.decrypt(nonce, package["ciphertext"], None)
        return plaintext.decode()


def exercise_3_key_exchange():
    """Simulate a secure chat between Alice and Bob."""
    alice = SecureChatParticipant(name="Alice")
    bob = SecureChatParticipant(name="Bob")

    # Key exchange
    alice.derive_keys(bob.public_key, is_initiator=True)
    bob.derive_keys(alice.public_key, is_initiator=False)

    # Alice sends messages to Bob
    msg1 = alice.encrypt_message("Hello Bob!")
    msg2 = alice.encrypt_message("How are you?")

    print(f"Bob receives: {bob.decrypt_message(msg1)}")
    print(f"Bob receives: {bob.decrypt_message(msg2)}")

    # Bob replies
    reply = bob.encrypt_message("I'm fine, thanks!")
    print(f"Alice receives: {alice.decrypt_message(reply)}")

    # Replay attack test
    try:
        bob.decrypt_message(msg1)  # Replay of msg1
        print("ERROR: Replay not detected!")
    except ValueError as e:
        print(f"Replay blocked: {e}")

    print("Key exchange and secure chat verified successfully.")


# ---------------------------------------------------------------------------
# Exercise 4: Nonce Reuse Attack (Explanation Only)
# ---------------------------------------------------------------------------

def exercise_4_nonce_reuse():
    """
    Demonstrate why nonce reuse is dangerous with AES-CTR.
    Given ct1, ct2 encrypted with same key+nonce, and knowing pt1,
    recover pt2.

    Mathematical explanation:
        ct1 = pt1 XOR keystream
        ct2 = pt2 XOR keystream
        ct1 XOR ct2 = pt1 XOR pt2  (keystream cancels out)
        pt2 = ct1 XOR ct2 XOR pt1
    """
    ct1 = bytes.fromhex("a1b2c3d4e5f6071829")
    ct2 = bytes.fromhex("b4a3d2c5f4e7162738")
    pt1 = b"plaintext"

    # Recovery: pt2 = ct1 XOR ct2 XOR pt1
    pt2 = bytes(a ^ b ^ c for a, b, c in zip(ct1, ct2, pt1))
    print(f"Recovered plaintext2: {pt2}")
    print(f"  (as hex): {pt2.hex()}")

    print("\nWhy this works:")
    print("  When two messages are encrypted with the same key and nonce,")
    print("  the keystream is identical. XOR-ing the two ciphertexts cancels")
    print("  out the keystream, leaving pt1 XOR pt2. Knowing pt1 gives us pt2.")

    print("\nPrevention:")
    print("  - Never reuse a nonce with the same key")
    print("  - Use AES-GCM (which detects nonce reuse via authentication tag)")
    print("  - Use random nonces with a large nonce space (e.g., XChaCha20)")
    print("  - Use key-committing AEAD constructions for maximum safety")


# ---------------------------------------------------------------------------
# Exercise 5: Digital Signature Verification (Code Signing)
# ---------------------------------------------------------------------------

@dataclass
class TrustedKey:
    """A trusted public key in the registry."""
    key_id: str
    public_key: ed25519.Ed25519PublicKey
    added_at: float
    revoked: bool = False


class CodeSigningSystem:
    """
    Simple code-signing system using Ed25519.
    A publisher signs Python scripts; a runner verifies before executing.
    """

    def __init__(self):
        self.trusted_keys: dict[str, TrustedKey] = {}
        self.max_signature_age_seconds: int = 30 * 24 * 3600  # 30 days

    def generate_signing_key(self, key_id: str):
        """Generate a new Ed25519 signing key pair."""
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        self.trusted_keys[key_id] = TrustedKey(
            key_id=key_id,
            public_key=public_key,
            added_at=time.time(),
        )
        return private_key, public_key

    def sign_script(self, script_content: bytes, private_key, key_id: str) -> dict:
        """Sign a script and return the signature package."""
        timestamp = int(time.time())
        # Sign: content + timestamp to bind signature to time
        data_to_sign = script_content + timestamp.to_bytes(8, "big")
        signature = private_key.sign(data_to_sign)
        return {
            "content": script_content,
            "signature": signature,
            "key_id": key_id,
            "timestamp": timestamp,
        }

    def verify_and_run(self, package: dict) -> bool:
        """Verify a signed script and run it if valid."""
        key_id = package["key_id"]

        # Check if key is trusted
        if key_id not in self.trusted_keys:
            print(f"  REJECTED: Unknown key '{key_id}'")
            return False

        trusted = self.trusted_keys[key_id]
        if trusted.revoked:
            print(f"  REJECTED: Key '{key_id}' has been revoked")
            return False

        # Check timestamp (reject signatures older than 30 days)
        age = time.time() - package["timestamp"]
        if age > self.max_signature_age_seconds:
            print(f"  REJECTED: Signature is {age / 86400:.0f} days old (max 30)")
            return False

        # Verify signature
        data_to_sign = package["content"] + package["timestamp"].to_bytes(8, "big")
        try:
            trusted.public_key.verify(package["signature"], data_to_sign)
            print(f"  VERIFIED: Script signed by '{key_id}' is authentic")
            return True
        except Exception:
            print(f"  REJECTED: Invalid signature for key '{key_id}'")
            return False

    def rotate_key(self, old_key_id: str, new_key_id: str):
        """Rotate a signing key — old key remains valid for existing signatures."""
        if old_key_id in self.trusted_keys:
            # Do NOT revoke old key — existing signatures should remain valid
            print(f"  Key '{old_key_id}' preserved for existing signatures")
        new_priv, new_pub = self.generate_signing_key(new_key_id)
        print(f"  New key '{new_key_id}' added to trusted registry")
        return new_priv


def exercise_5_code_signing():
    """Demonstrate a code-signing system with Ed25519."""
    system = CodeSigningSystem()

    # Publisher generates a key
    priv_key, pub_key = system.generate_signing_key("publisher-v1")

    # Publisher signs a script
    script = b'print("Hello, verified world!")'
    package = system.sign_script(script, priv_key, "publisher-v1")
    print("Verifying signed script:")
    assert system.verify_and_run(package)

    # Tampered script
    tampered_package = dict(package)
    tampered_package["content"] = b'print("TAMPERED!")'
    print("\nVerifying tampered script:")
    system.verify_and_run(tampered_package)

    # Key rotation
    print("\nRotating key:")
    new_priv = system.rotate_key("publisher-v1", "publisher-v2")

    # Old signature still works with old key
    print("\nVerifying old script after key rotation:")
    assert system.verify_and_run(package)

    print("\nCode signing system verified successfully.")


# ---------------------------------------------------------------------------
# Exercise 6: Cryptographic Audit
# ---------------------------------------------------------------------------

def exercise_6_cryptographic_audit():
    """
    Identify all cryptographic vulnerabilities in the given code
    and explain each issue with its fix.
    """
    vulnerabilities = [
        {
            "issue": "MD5 used for key derivation",
            "danger": "MD5 is broken (collisions in seconds). It also produces "
                      "only a 128-bit key, below the 256-bit AES recommendation.",
            "fix": "Use PBKDF2, scrypt, or Argon2 to derive keys from passwords, "
                   "or use a proper KDF like HKDF for random input.",
        },
        {
            "issue": "Static IV (all zeros)",
            "danger": "Reusing the same IV with the same key enables attacks "
                      "against CBC mode (reveals plaintext patterns).",
            "fix": "Generate a random IV for each encryption operation using "
                   "os.urandom(16).",
        },
        {
            "issue": "CBC mode without authentication (no HMAC/tag)",
            "danger": "CBC is malleable — an attacker can modify ciphertext and "
                      "predictably alter the plaintext (padding oracle attack).",
            "fix": "Use AES-GCM or ChaCha20-Poly1305 which provide "
                   "authenticated encryption.",
        },
        {
            "issue": "Manual padding implementation",
            "danger": "Custom padding is error-prone and may introduce padding "
                      "oracle vulnerabilities.",
            "fix": "Use an AEAD cipher (GCM/Poly1305) that handles padding "
                   "internally, or use cryptography library's padding module.",
        },
        {
            "issue": "Password-to-key: no salt",
            "danger": "Without a unique salt, identical passwords produce "
                      "identical keys, enabling rainbow table attacks.",
            "fix": "Use a random 16+ byte salt stored alongside the ciphertext.",
        },
        {
            "issue": "Password-to-key: no key stretching (iterations)",
            "danger": "A single MD5 hash is near-instant — brute-force is trivial.",
            "fix": "Use PBKDF2 with 600,000+ iterations, or scrypt/Argon2.",
        },
        {
            "issue": "SHA-256 for password verification (verify_password)",
            "danger": "SHA-256 is a fast hash, not suitable for password storage. "
                      "Enables GPU-accelerated brute-force.",
            "fix": "Use bcrypt, scrypt, or Argon2id for password hashing.",
        },
        {
            "issue": "Timing-vulnerable comparison (== operator)",
            "danger": "String equality (==) short-circuits on first mismatch, "
                      "leaking information about the hash via timing.",
            "fix": "Use hmac.compare_digest() for constant-time comparison.",
        },
        {
            "issue": "No password salt in verify_password",
            "danger": "Identical passwords produce identical hashes, enabling "
                      "precomputed dictionary attacks.",
            "fix": "Store a unique salt per user; hash = H(salt + password).",
        },
    ]

    print("Cryptographic Audit — Vulnerabilities Found")
    print("=" * 60)
    for i, v in enumerate(vulnerabilities, 1):
        print(f"\n{i}. {v['issue']}")
        print(f"   Danger: {v['danger']}")
        print(f"   Fix:    {v['fix']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Exercise 1: Symmetric Encryption (AES-GCM)")
    print("=" * 70)
    exercise_1_symmetric_encryption()

    print("\n" + "=" * 70)
    print("Exercise 2: Hybrid Encryption (RSA + AES-GCM)")
    print("=" * 70)
    exercise_2_hybrid_encryption()

    print("\n" + "=" * 70)
    print("Exercise 3: Key Exchange Protocol (X25519 + ChaCha20)")
    print("=" * 70)
    exercise_3_key_exchange()

    print("\n" + "=" * 70)
    print("Exercise 4: Nonce Reuse Attack (Explanation)")
    print("=" * 70)
    exercise_4_nonce_reuse()

    print("\n" + "=" * 70)
    print("Exercise 5: Code Signing System (Ed25519)")
    print("=" * 70)
    exercise_5_code_signing()

    print("\n" + "=" * 70)
    print("Exercise 6: Cryptographic Audit")
    print("=" * 70)
    exercise_6_cryptographic_audit()
