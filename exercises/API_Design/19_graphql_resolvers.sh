#!/bin/bash
# Exercises for Lesson 19: GraphQL Resolvers
# Topic: API_Design
# Solutions to practice problems from the lesson.

# === Exercise 1: DataLoader Implementation ===
exercise_1() {
    echo "=== Exercise 1: DataLoader Implementation ==="
    cat << 'SOLUTION'
from strawberry.dataloader import DataLoader
from typing import Optional

# Batch function: load customers by IDs
async def load_customers(keys: list[str]) -> list[Optional["Customer"]]:
    customers = await db.query(Customer).filter(Customer.id.in_(keys)).all()
    customer_map = {str(c.id): c for c in customers}
    return [customer_map.get(key) for key in keys]

# Batch function: load order items grouped by order ID
async def load_items_by_order(keys: list[str]) -> list[list["OrderItem"]]:
    items = await db.query(OrderItem).filter(OrderItem.order_id.in_(keys)).all()
    items_by_order: dict[str, list[OrderItem]] = {}
    for item in items:
        items_by_order.setdefault(str(item.order_id), []).append(item)
    return [items_by_order.get(key, []) for key in keys]

# Batch function: load products by IDs
async def load_products(keys: list[str]) -> list[Optional["Product"]]:
    products = await db.query(Product).filter(Product.id.in_(keys)).all()
    product_map = {str(p.id): p for p in products}
    return [product_map.get(key) for key in keys]

# DataLoaders container
@dataclass
class DataLoaders:
    def __init__(self, db):
        self.customer_loader = DataLoader(load_fn=load_customers)
        self.items_by_order_loader = DataLoader(load_fn=load_items_by_order)
        self.product_loader = DataLoader(load_fn=load_products)

# Usage in resolvers
@strawberry.type
class Order:
    id: strawberry.ID
    customer_id: strawberry.ID

    @strawberry.field
    async def customer(self, info) -> "Customer":
        return await info.context.dataloaders.customer_loader.load(self.customer_id)

    @strawberry.field
    async def items(self, info) -> list["OrderItem"]:
        return await info.context.dataloaders.items_by_order_loader.load(self.id)

@strawberry.type
class OrderItem:
    product_id: strawberry.ID
    quantity: int

    @strawberry.field
    async def product(self, info) -> "Product":
        return await info.context.dataloaders.product_loader.load(self.product_id)
SOLUTION
}

# === Exercise 2: Authorization Resolver ===
exercise_2() {
    echo "=== Exercise 2: Authorization Resolver ==="
    cat << 'SOLUTION'
@strawberry.type
class User:
    id: strawberry.ID
    username: str
    _email: strawberry.Private[str]
    _email_public: strawberry.Private[bool]

    @strawberry.field
    def email(self, info: strawberry.types.Info) -> str | None:
        viewer = info.context.current_user

        # Case 1: viewer is the user themselves
        if viewer and viewer.id == self.id:
            return self._email

        # Case 2: viewer has ADMIN role
        if viewer and viewer.role == "ADMIN":
            return self._email

        # Case 3: user's emailPublic setting is true
        if self._email_public:
            return self._email

        # Otherwise: return null
        return None
SOLUTION
}

# === Exercise 3: Error Handling ===
exercise_3() {
    echo "=== Exercise 3: Transfer Funds Error Handling ==="
    cat << 'SOLUTION'
@strawberry.input
class TransferFundsInput:
    source_account_id: strawberry.ID
    destination_account_id: strawberry.ID
    amount: float
    currency: str

@strawberry.type
class TransferFundsPayload:
    transfer: "Transfer | None" = None
    user_errors: list["UserError"] = strawberry.field(default_factory=list)

@strawberry.type
class Mutation:
    @strawberry.mutation
    async def transfer_funds(
        self, info, input: TransferFundsInput
    ) -> TransferFundsPayload:
        errors = []

        # Validate: self-transfer
        if input.source_account_id == input.destination_account_id:
            errors.append(UserError(
                field=["input", "destinationAccountId"],
                message="Cannot transfer to the same account",
                code="SELF_TRANSFER",
            ))
            return TransferFundsPayload(user_errors=errors)

        # Validate: source account exists
        source = await account_repo.find_by_id(input.source_account_id)
        if source is None:
            errors.append(UserError(
                field=["input", "sourceAccountId"],
                message="Source account not found",
                code="NOT_FOUND",
            ))

        # Validate: destination account exists
        dest = await account_repo.find_by_id(input.destination_account_id)
        if dest is None:
            errors.append(UserError(
                field=["input", "destinationAccountId"],
                message="Destination account not found",
                code="NOT_FOUND",
            ))

        if errors:
            return TransferFundsPayload(user_errors=errors)

        # Validate: sufficient balance
        if source.balance < input.amount:
            errors.append(UserError(
                field=["input", "amount"],
                message=f"Insufficient balance. Available: {source.balance}",
                code="INSUFFICIENT_FUNDS",
            ))
            return TransferFundsPayload(user_errors=errors)

        # Execute transfer
        transfer = await transfer_service.execute(
            source_id=input.source_account_id,
            dest_id=input.destination_account_id,
            amount=input.amount,
        )
        return TransferFundsPayload(transfer=transfer)
SOLUTION
}

# === Exercise 4: Service Layer Refactoring ===
exercise_4() {
    echo "=== Exercise 4: Service Layer Refactoring ==="
    cat << 'SOLUTION'
# Service layer
class OrderService:
    def __init__(self, db, current_user):
        self.db = db
        self.current_user = current_user

    async def create_order(self, input) -> CreateOrderPayload:
        if not self.current_user:
            return CreateOrderPayload(user_errors=[
                UserError(field=[], message="Authentication required", code="UNAUTHENTICATED")
            ])

        errors = []
        items = []
        total = 0

        for item_input in input.items:
            product = await self.db.query(Product).get(item_input.product_id)
            if not product:
                errors.append(UserError(
                    field=["input", "items", "productId"],
                    message=f"Product {item_input.product_id} not found",
                    code="NOT_FOUND",
                ))
                continue
            if product.stock < item_input.quantity:
                errors.append(UserError(
                    field=["input", "items", "quantity"],
                    message=f"Insufficient stock for {product.name}",
                    code="INSUFFICIENT_STOCK",
                ))
                continue
            items.append(OrderItem(product=product, quantity=item_input.quantity))
            total += product.price * item_input.quantity

        if errors:
            return CreateOrderPayload(user_errors=errors)

        order = Order(user_id=self.current_user.id, items=items, total=total)
        self.db.add(order)
        await self.db.commit()
        return CreateOrderPayload(order=order)

# Thin resolver
@strawberry.type
class Mutation:
    @strawberry.mutation
    async def create_order(self, info, input) -> CreateOrderPayload:
        service = OrderService(info.context.db, info.context.current_user)
        return await service.create_order(input)
SOLUTION
}

main() {
    exercise_1; echo ""; exercise_2; echo ""; exercise_3; echo ""; exercise_4
}
main "$@"
