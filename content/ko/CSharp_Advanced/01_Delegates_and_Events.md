# 델리게이트와 이벤트

**이전**: [개요](./00_Overview.md) | **다음**: [람다 표현식과 클로저](./02_Lambda_and_Closures.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 메서드를 참조하기 위한 델리게이트(delegate)를 선언하고 인스턴스화하기
2. 멀티캐스트 델리게이트 체이닝을 사용하여 델리게이트를 결합하고 제거하기
3. 내장 델리게이트 타입 `Action`, `Func`, `Predicate` 사용하기
4. `EventHandler<T>`로 이벤트를 선언하고 안전하게 발생시키기
5. 도메인별 이벤트를 위한 사용자 정의 `EventArgs` 서브클래스 설계하기
6. C#에서 발행자-구독자(publisher-subscriber) 패턴 구현하기
7. `?.Invoke()`를 사용하여 스레드 안전한 방식으로 이벤트 호출하기

---

델리게이트(delegate)는 C#에서 콜백 기반 프로그래밍의 기초입니다. 델리게이트를 사용하면 메서드를 일급 값으로 취급할 수 있습니다 — 변수에 저장하고, 인자로 전달하고, 체인으로 조합할 수 있습니다. 이벤트(event)는 델리게이트 위에 구축되어 안전하고 캡슐화된 알림 메커니즘을 제공하며, .NET 생태계 전반의 UI 프레임워크, 메시지 버스, 반응형 아키텍처의 핵심입니다.

## 1. 델리게이트 기초

델리게이트는 특정 매개변수 목록과 반환 타입을 가진 메서드에 대한 참조를 나타내는 타입입니다. 타입 안전한 함수 포인터(function pointer)라고 생각하면 됩니다.

### 1.1 델리게이트 타입 선언

델리게이트 선언은 새로운 타입을 정의합니다. 시그니처가 델리게이트와 일치하는 모든 메서드를 해당 델리게이트의 인스턴스에 할당할 수 있습니다.

```csharp
// 두 개의 int를 받아 int를 반환하는 델리게이트 타입 선언
public delegate int MathOperation(int a, int b);

// 문자열을 받아 void를 반환하는 델리게이트 타입 선언
public delegate void Logger(string message);

// 매개변수 없이 반환도 없는 델리게이트 타입 선언
public delegate void SimpleCallback();
```

### 1.2 델리게이트 인스턴스화

일치하는 메서드 이름을 생성자에 전달하거나 메서드 그룹(method group)을 직접 할당하여 델리게이트 인스턴스를 생성할 수 있습니다.

```csharp
public class Calculator
{
    public static int Add(int a, int b) => a + b;
    public static int Multiply(int a, int b) => a * b;
    public int Subtract(int a, int b) => a - b; // 인스턴스 메서드
}

// 생성자 구문
MathOperation op1 = new MathOperation(Calculator.Add);
Console.WriteLine(op1(3, 4)); // 7

// 메서드 그룹 변환 (선호 — 더 간결함)
MathOperation op2 = Calculator.Multiply;
Console.WriteLine(op2(3, 4)); // 12

// 인스턴스 메서드 델리게이트
var calc = new Calculator();
MathOperation op3 = calc.Subtract;
Console.WriteLine(op3(10, 4)); // 6
```

### 1.3 델리게이트 호출

델리게이트를 일반 메서드 호출처럼 호출할 수 있습니다. 또는 `Invoke` 메서드를 명시적으로 사용할 수도 있습니다.

```csharp
MathOperation op = Calculator.Add;

// 직접 호출
int result1 = op(5, 3); // 8

// 명시적 Invoke
int result2 = op.Invoke(5, 3); // 8
```

### 1.4 델리게이트 분산

델리게이트는 반환 타입에 대한 공변성(covariance)과 매개변수 타입에 대한 반공변성(contravariance)을 지원합니다. 이는 델리게이트가 반환 타입이 더 파생된 메서드나 매개변수 타입이 덜 파생된 메서드를 참조할 수 있다는 것을 의미합니다.

```csharp
public delegate Animal AnimalFactory();
public delegate void ProcessDog(Dog dog);

public class Animal { }
public class Dog : Animal { }

public static Dog CreateDog() => new Dog();
public static void ProcessAnimal(Animal animal) { }

// 공변성: Dog은 Animal보다 더 파생됨
AnimalFactory factory = CreateDog;  // OK — 반환 타입 공변성

// 반공변성: Animal은 Dog보다 덜 파생됨
ProcessDog processor = ProcessAnimal; // OK — 매개변수 반공변성
```

## 2. 멀티캐스트 델리게이트

C#의 모든 델리게이트는 멀티캐스트입니다 — 하나 이상의 메서드에 대한 참조를 보유할 수 있습니다. 멀티캐스트 델리게이트가 호출되면 호출 목록의 모든 메서드가 순서대로 호출됩니다.

### 2.1 +=와 -=로 델리게이트 결합

```csharp
public delegate void Notifier(string message);

public static void EmailNotify(string msg) =>
    Console.WriteLine($"[EMAIL] {msg}");

public static void SmsNotify(string msg) =>
    Console.WriteLine($"[SMS] {msg}");

public static void SlackNotify(string msg) =>
    Console.WriteLine($"[SLACK] {msg}");

// 델리게이트 결합
Notifier notify = EmailNotify;
notify += SmsNotify;
notify += SlackNotify;

notify("서버가 다운되었습니다!");
// 출력:
// [EMAIL] 서버가 다운되었습니다!
// [SMS] 서버가 다운되었습니다!
// [SLACK] 서버가 다운되었습니다!

// 델리게이트 제거
notify -= SmsNotify;
notify("서버가 복구되었습니다.");
// 출력:
// [EMAIL] 서버가 복구되었습니다.
// [SLACK] 서버가 복구되었습니다.
```

### 2.2 멀티캐스트 델리게이트의 반환 값

멀티캐스트 델리게이트가 void가 아닌 반환 타입을 가질 때, 호출 목록의 **마지막** 메서드의 반환 값만 반환됩니다. 모든 반환 값이 필요하면 호출 목록을 수동으로 순회하세요.

```csharp
public delegate int Scorer(string input);

public static int LengthScore(string s) => s.Length;
public static int VowelScore(string s) => s.Count(c => "aeiouAEIOU".Contains(c));

Scorer scorer = LengthScore;
scorer += VowelScore;

// VowelScore의 반환 값만 캡처됨
int result = scorer("Hello"); // 2 (모음: e, o)

// 모든 결과를 얻으려면 호출 목록을 순회
foreach (Scorer s in scorer.GetInvocationList().Cast<Scorer>())
{
    Console.WriteLine(s("Hello")); // 5, 그 다음 2
}
```

### 2.3 Delegate.Combine과 Delegate.Remove

`+=`와 `-=` 연산자는 `Delegate.Combine`과 `Delegate.Remove`의 구문 설탕(syntactic sugar)입니다.

```csharp
Notifier a = EmailNotify;
Notifier b = SmsNotify;

// 다음은 동일함:
Notifier combined1 = a + b;
Notifier combined2 = (Notifier)Delegate.Combine(a, b);

// 제거:
Notifier reduced1 = combined1 - b;
Notifier reduced2 = (Notifier)Delegate.Remove(combined2, b);
```

## 3. 내장 델리게이트 타입

.NET 기본 클래스 라이브러리는 대부분의 사용 사례를 커버하는 제네릭 델리게이트 타입을 제공하므로, 사용자 정의 델리게이트 타입을 선언할 필요가 거의 없습니다.

### 3.1 Action\<T\> — Void 반환

`Action`은 `void`를 반환하는 메서드를 래핑합니다. 0개에서 16개까지의 타입 매개변수 변형이 있습니다.

```csharp
// 매개변수 없음
Action greet = () => Console.WriteLine("Hello!");
greet();

// 매개변수 하나
Action<string> log = message => Console.WriteLine($"[LOG] {message}");
log("애플리케이션 시작");

// 매개변수 두 개
Action<string, int> repeat = (text, count) =>
{
    for (int i = 0; i < count; i++)
        Console.Write(text);
    Console.WriteLine();
};
repeat("Ha", 3); // HaHaHa

// 메서드 매개변수로 — 전략 패턴
public static void ProcessItems<T>(IEnumerable<T> items, Action<T> processor)
{
    foreach (var item in items)
        processor(item);
}

ProcessItems(new[] { 1, 2, 3 }, n => Console.WriteLine(n * 10));
// 10, 20, 30
```

### 3.2 Func\<T, TResult\> — 반환 값 포함

`Func`은 값을 반환하는 메서드를 래핑합니다. 마지막 타입 매개변수가 항상 반환 타입입니다.

```csharp
// 입력 없음, string 반환
Func<string> getName = () => "Alice";
Console.WriteLine(getName()); // Alice

// int -> bool
Func<int, bool> isEven = n => n % 2 == 0;
Console.WriteLine(isEven(4)); // True

// (int, int) -> int
Func<int, int, int> add = (a, b) => a + b;
Console.WriteLine(add(3, 7)); // 10

// Func을 팩토리/지연 초기화기로 사용
public static T CreateOrDefault<T>(Func<T> factory, bool shouldCreate)
{
    return shouldCreate ? factory() : default!;
}

var result = CreateOrDefault(() => new List<int> { 1, 2, 3 }, true);
Console.WriteLine(result.Count); // 3
```

### 3.3 Predicate\<T\> — 불리언 테스트

`Predicate<T>`는 `Func<T, bool>`과 동일하지만, `List<T>.Find`, `List<T>.RemoveAll` 등 이전 컬렉션 API에서 특별히 사용됩니다.

```csharp
Predicate<int> isPositive = n => n > 0;

var numbers = new List<int> { -3, -1, 0, 2, 5, -7, 8 };

// List<T>.FindAll은 Predicate<T>를 사용
List<int> positives = numbers.FindAll(isPositive);
Console.WriteLine(string.Join(", ", positives)); // 2, 5, 8

// List<T>.RemoveAll은 Predicate<T>를 사용
int removed = numbers.RemoveAll(n => n < 0);
Console.WriteLine(removed); // 3
Console.WriteLine(string.Join(", ", numbers)); // 0, 2, 5, 8
```

## 4. 익명 메서드

람다 표현식(C# 3.0) 이전에, C# 2.0에서는 `delegate` 키워드를 사용한 익명 메서드(anonymous method)를 도입했습니다. 대부분 람다로 대체되었지만 레거시 코드에서 여전히 나타납니다.

### 4.1 익명 메서드 구문

```csharp
// delegate 키워드를 사용한 익명 메서드
Func<int, int, int> multiply = delegate (int a, int b)
{
    return a * b;
};
Console.WriteLine(multiply(4, 5)); // 20

// 매개변수를 무시하는 익명 메서드
// (매개변수 목록 없는 delegate는 모든 인자를 버림)
EventHandler handler = delegate
{
    Console.WriteLine("무언가 일어났지만, 세부 사항은 신경 쓰지 않습니다.");
};
handler(null, EventArgs.Empty);
```

### 4.2 익명 메서드를 선호할 때

현대 C#에서는 람다 대신 `delegate` 익명 메서드를 사용할 이유가 거의 없습니다. 유일한 예외는 모든 매개변수를 정말로 무시하고 싶을 때입니다:

```csharp
// 람다는 디스카드나 매개변수를 지정해야 함
button.Click += (_, _) => HandleClick();

// 익명 메서드는 완전히 생략 가능
button.Click += delegate { HandleClick(); };
```

## 5. 이벤트와 EventHandler

이벤트(event)는 델리게이트 위에 캡슐화 계층을 제공합니다. 델리게이트 필드는 접근 가능한 모든 코드에서 호출하거나 재할당할 수 있지만, 이벤트는 외부 코드를 `+=`와 `-=` 연산만으로 제한합니다.

### 5.1 이벤트 선언과 발생

```csharp
public class TemperatureSensor
{
    // EventHandler<T>를 사용한 이벤트 선언
    public event EventHandler<TemperatureChangedEventArgs>? TemperatureChanged;

    private double _temperature;

    public double Temperature
    {
        get => _temperature;
        set
        {
            double oldTemp = _temperature;
            _temperature = value;
            OnTemperatureChanged(new TemperatureChangedEventArgs(oldTemp, value));
        }
    }

    // 이벤트를 발생시키기 위한 protected virtual 메서드 (표준 패턴)
    protected virtual void OnTemperatureChanged(TemperatureChangedEventArgs e)
    {
        TemperatureChanged?.Invoke(this, e);
    }
}
```

### 5.2 사용자 정의 EventArgs

```csharp
public class TemperatureChangedEventArgs : EventArgs
{
    public double OldTemperature { get; }
    public double NewTemperature { get; }
    public double Delta => NewTemperature - OldTemperature;

    public TemperatureChangedEventArgs(double oldTemp, double newTemp)
    {
        OldTemperature = oldTemp;
        NewTemperature = newTemp;
    }
}
```

### 5.3 이벤트 구독

```csharp
var sensor = new TemperatureSensor();

// 메서드 그룹으로 구독
sensor.TemperatureChanged += OnTemperatureChanged;

// 람다로 구독
sensor.TemperatureChanged += (sender, e) =>
{
    if (e.Delta > 5)
        Console.WriteLine($"경고: {e.Delta:F1}°C의 온도 급등!");
};

sensor.Temperature = 20.0;
sensor.Temperature = 28.5; // 경고 트리거 (delta = 8.5)

static void OnTemperatureChanged(object? sender, TemperatureChangedEventArgs e)
{
    Console.WriteLine($"온도: {e.OldTemperature:F1} -> {e.NewTemperature:F1}");
}
```

## 6. 이벤트 선언 패턴

### 6.1 EventHandler를 사용한 간단한 이벤트

추가 데이터를 전달하지 않는 이벤트에는 비제네릭 `EventHandler`를 사용합니다.

```csharp
public class Button
{
    public event EventHandler? Clicked;

    public void SimulateClick()
    {
        Console.WriteLine("버튼이 클릭되었습니다.");
        Clicked?.Invoke(this, EventArgs.Empty);
    }
}

var btn = new Button();
btn.Clicked += (sender, e) => Console.WriteLine("핸들러 1 실행됨");
btn.Clicked += (sender, e) => Console.WriteLine("핸들러 2 실행됨");
btn.SimulateClick();
// 버튼이 클릭되었습니다.
// 핸들러 1 실행됨
// 핸들러 2 실행됨
```

### 6.2 사용자 정의 이벤트 접근자 (add/remove)

로깅이나 스레드 동기화 같은 이벤트 구독의 세밀한 제어를 위해 명시적 `add`와 `remove` 접근자를 제공할 수 있습니다.

```csharp
public class SecurePublisher
{
    private EventHandler? _completed;
    private readonly object _lock = new();

    public event EventHandler Completed
    {
        add
        {
            lock (_lock)
            {
                Console.WriteLine($"구독자 추가됨: {value.Method.Name}");
                _completed += value;
            }
        }
        remove
        {
            lock (_lock)
            {
                Console.WriteLine($"구독자 제거됨: {value.Method.Name}");
                _completed -= value;
            }
        }
    }

    public void RaiseCompleted()
    {
        EventHandler? handler;
        lock (_lock)
        {
            handler = _completed;
        }
        handler?.Invoke(this, EventArgs.Empty);
    }
}
```

### 6.3 이벤트 vs 델리게이트 — 접근 제어

이벤트와 델리게이트 필드의 핵심 차이점은 캡슐화입니다:

```csharp
public class WithDelegateField
{
    public Action<string>? OnMessage; // public 델리게이트 필드
}

public class WithEvent
{
    public event Action<string>? OnMessage; // 이벤트

    public void Send(string msg) => OnMessage?.Invoke(msg);
}

var df = new WithDelegateField();
df.OnMessage = msg => Console.WriteLine(msg); // OK — 전체 할당
df.OnMessage("test");        // OK — 외부 호출
df.OnMessage = null;         // OK — 외부에서 모든 구독자를 지울 수 있음

var ev = new WithEvent();
ev.OnMessage += msg => Console.WriteLine(msg); // OK — 구독
// ev.OnMessage("test");     // 오류 — 외부에서 호출 불가
// ev.OnMessage = null;      // 오류 — 외부에서 할당 불가
ev.Send("test");             // OK — 클래스 자체의 메서드를 통해야 함
```

## 7. 스레드 안전한 이벤트 호출

### 7.1 Null 조건부 패턴

이벤트의 고전적인 함정은 경쟁 조건(race condition)입니다: null 검사와 호출 사이에 다른 스레드가 마지막 핸들러의 구독을 취소할 수 있습니다. null 조건부 연산자 `?.`는 델리게이트 참조를 원자적으로 캡처하여 이 문제를 해결합니다.

```csharp
// 잘못됨 — 경쟁 조건 가능
if (TemperatureChanged != null)       // 여기서 다른 스레드가 -=를 실행
    TemperatureChanged(this, args);   // NullReferenceException!

// 올바름 — null 조건부 연산자
TemperatureChanged?.Invoke(this, args);

// 역시 올바름 — 명시적 로컬 복사
var handler = TemperatureChanged;
handler?.Invoke(this, args);
```

### 7.2 Volatile 델리게이트 읽기

동시 구독과 호출이 빈번한 클래스에서는 `Volatile.Read`를 사용하여 델리게이트 참조가 최신인지 확인할 수 있습니다:

```csharp
protected virtual void OnDataReceived(DataReceivedEventArgs e)
{
    var handler = Volatile.Read(ref _dataReceived);
    handler?.Invoke(this, e);
}
```

## 8. 발행자-구독자 패턴

발행자-구독자(pub-sub) 패턴은 이벤트 생산자와 소비자를 분리합니다. C# 이벤트는 이 패턴의 자연스러운 구현입니다.

### 8.1 완전한 Pub-Sub 예제: 주식 시세

```csharp
// 이벤트 인자
public class StockPriceChangedEventArgs : EventArgs
{
    public string Symbol { get; }
    public decimal OldPrice { get; }
    public decimal NewPrice { get; }
    public decimal ChangePercent => OldPrice == 0 ? 0 :
        Math.Round((NewPrice - OldPrice) / OldPrice * 100, 2);

    public StockPriceChangedEventArgs(string symbol, decimal oldPrice, decimal newPrice)
    {
        Symbol = symbol;
        OldPrice = oldPrice;
        NewPrice = newPrice;
    }
}

// 발행자
public class StockTicker
{
    private readonly Dictionary<string, decimal> _prices = new();
    public event EventHandler<StockPriceChangedEventArgs>? PriceChanged;

    public void UpdatePrice(string symbol, decimal newPrice)
    {
        _prices.TryGetValue(symbol, out decimal oldPrice);
        _prices[symbol] = newPrice;

        if (oldPrice != newPrice)
        {
            PriceChanged?.Invoke(this,
                new StockPriceChangedEventArgs(symbol, oldPrice, newPrice));
        }
    }
}

// 구독자 1: 콘솔 로거
public class PriceLogger
{
    public void Subscribe(StockTicker ticker)
    {
        ticker.PriceChanged += OnPriceChanged;
    }

    private void OnPriceChanged(object? sender, StockPriceChangedEventArgs e)
    {
        Console.WriteLine(
            $"[LOG] {e.Symbol}: ${e.OldPrice} -> ${e.NewPrice} ({e.ChangePercent:+0.00;-0.00}%)");
    }
}

// 구독자 2: 알림 시스템
public class PriceAlert
{
    private readonly decimal _thresholdPercent;

    public PriceAlert(decimal thresholdPercent) => _thresholdPercent = thresholdPercent;

    public void Subscribe(StockTicker ticker)
    {
        ticker.PriceChanged += (_, e) =>
        {
            if (Math.Abs(e.ChangePercent) >= _thresholdPercent)
            {
                Console.WriteLine(
                    $"*** 경고: {e.Symbol}이(가) {e.ChangePercent:+0.00;-0.00}% 변동했습니다 ***");
            }
        };
    }
}

// 사용법
var ticker = new StockTicker();
var logger = new PriceLogger();
var alert = new PriceAlert(thresholdPercent: 3.0m);

logger.Subscribe(ticker);
alert.Subscribe(ticker);

ticker.UpdatePrice("MSFT", 350.00m);
ticker.UpdatePrice("MSFT", 365.00m); // +4.29% — 경고 트리거
ticker.UpdatePrice("AAPL", 180.00m);
ticker.UpdatePrice("AAPL", 178.50m); // -0.83% — 경고 없음
```

## 9. 실전 예제: 이벤트 기반 알림 시스템

여러 이벤트 타입, 구독 취소, 약한 이벤트 유사 패턴을 보여주는 더 완전한 이벤트 기반 알림 시스템을 만들어 보겠습니다.

### 9.1 도메인 모델

```csharp
public enum OrderStatus { Created, Processing, Shipped, Delivered, Cancelled }

public class OrderStatusChangedEventArgs : EventArgs
{
    public int OrderId { get; }
    public OrderStatus OldStatus { get; }
    public OrderStatus NewStatus { get; }
    public DateTime Timestamp { get; }

    public OrderStatusChangedEventArgs(int orderId, OrderStatus oldStatus, OrderStatus newStatus)
    {
        OrderId = orderId;
        OldStatus = oldStatus;
        NewStatus = newStatus;
        Timestamp = DateTime.UtcNow;
    }
}
```

### 9.2 발행자 (Order)

```csharp
public class Order
{
    public int Id { get; }
    public string CustomerEmail { get; }

    private OrderStatus _status = OrderStatus.Created;
    public OrderStatus Status
    {
        get => _status;
        private set
        {
            if (_status != value)
            {
                var old = _status;
                _status = value;
                OnStatusChanged(new OrderStatusChangedEventArgs(Id, old, value));
            }
        }
    }

    public event EventHandler<OrderStatusChangedEventArgs>? StatusChanged;

    public Order(int id, string customerEmail)
    {
        Id = id;
        CustomerEmail = customerEmail;
    }

    public void Process() => Status = OrderStatus.Processing;
    public void Ship() => Status = OrderStatus.Shipped;
    public void Deliver() => Status = OrderStatus.Delivered;
    public void Cancel() => Status = OrderStatus.Cancelled;

    protected virtual void OnStatusChanged(OrderStatusChangedEventArgs e)
    {
        StatusChanged?.Invoke(this, e);
    }
}
```

### 9.3 구독자

```csharp
public class EmailNotificationService
{
    public void Subscribe(Order order) => order.StatusChanged += HandleStatusChange;
    public void Unsubscribe(Order order) => order.StatusChanged -= HandleStatusChange;

    private void HandleStatusChange(object? sender, OrderStatusChangedEventArgs e)
    {
        if (sender is Order order)
        {
            Console.WriteLine(
                $"[EMAIL -> {order.CustomerEmail}] " +
                $"주문 #{e.OrderId}: {e.OldStatus} -> {e.NewStatus}");
        }
    }
}

public class InventoryService
{
    public void Subscribe(Order order) => order.StatusChanged += HandleStatusChange;

    private void HandleStatusChange(object? sender, OrderStatusChangedEventArgs e)
    {
        switch (e.NewStatus)
        {
            case OrderStatus.Cancelled:
                Console.WriteLine($"[INVENTORY] 주문 #{e.OrderId}의 상품 재입고 중");
                break;
            case OrderStatus.Shipped:
                Console.WriteLine($"[INVENTORY] 주문 #{e.OrderId}의 상품 발송됨");
                break;
        }
    }
}

public class AuditLog
{
    private readonly List<string> _entries = new();

    public void Subscribe(Order order) => order.StatusChanged += HandleStatusChange;

    private void HandleStatusChange(object? sender, OrderStatusChangedEventArgs e)
    {
        string entry = $"[{e.Timestamp:u}] 주문 #{e.OrderId}: {e.OldStatus} -> {e.NewStatus}";
        _entries.Add(entry);
        Console.WriteLine($"[AUDIT] {entry}");
    }

    public IReadOnlyList<string> GetEntries() => _entries.AsReadOnly();
}
```

### 9.4 시스템 실행

```csharp
// 주문과 서비스 생성
var order = new Order(1001, "alice@example.com");
var emailService = new EmailNotificationService();
var inventoryService = new InventoryService();
var auditLog = new AuditLog();

// 모든 서비스 구독
emailService.Subscribe(order);
inventoryService.Subscribe(order);
auditLog.Subscribe(order);

// 주문 수명주기 처리
order.Process();   // Created -> Processing
order.Ship();      // Processing -> Shipped
order.Deliver();   // Shipped -> Delivered

Console.WriteLine($"\n감사 로그에 {auditLog.GetEntries().Count}개의 항목이 있습니다.");

// 다른 주문 생성 후 취소
var order2 = new Order(1002, "bob@example.com");
emailService.Subscribe(order2);
inventoryService.Subscribe(order2);
auditLog.Subscribe(order2);

order2.Process();
order2.Cancel(); // 재고 재입고 트리거
```

## 10. 연습 문제

1. **사용자 정의 델리게이트 체인**: `string`을 받아 `string`을 반환하는 델리게이트 `StringTransform`을 선언하세요. 세 가지 변환(공백 제거, 소문자 변환, 공백을 하이픈으로 대체)의 체인을 만드세요. `"  Hello Beautiful World  "`에 체인을 적용하고 호출 목록을 순회하며 각 중간 결과를 출력하세요.

2. **제네릭 이벤트 집약기**: `Subscribe<TEvent>(Action<TEvent> handler)`와 `Publish<TEvent>(TEvent eventData)` 메서드를 가진 `EventAggregator` 클래스를 구현하세요. 구독자는 자신이 구독한 타입의 이벤트만 받아야 합니다. 최소 두 가지 다른 이벤트 타입으로 테스트하세요.

3. **구독 취소와 메모리 누수**: 매초 `Tick` 이벤트를 발생시키는 `Timer` 클래스를 만드세요 (루프에서 `Task.Delay` 사용). 핸들러를 구독하고 틱을 관찰한 다음, 구독을 취소하고 틱 처리가 중지되는 것을 보여주세요. 구독을 취소하지 않으면 어떤 일이 일어날지 논의하세요.

4. **취소 가능 이벤트**: 구독자가 작업을 취소할 수 있는 이벤트 시스템을 설계하세요. `Cancel` 속성을 포함하는 사용자 정의 `DownloadingEventArgs`를 사용하여 `Downloading` 이벤트를 가진 `FileDownloader` 클래스를 만드세요. 다운로드 전에 이벤트를 발생시키고, 구독자가 `Cancel = true`를 설정하면 다운로드를 중단하세요.

5. **델리게이트 성능 비교**: (a) 직접 메서드 호출, (b) `Func<int, int>` 델리게이트, (c) 인터페이스 메서드 호출(`ITransform.Apply`)의 호출 비용을 비교하는 벤치마크를 작성하세요. 각각 1천만 번 실행하고 경과 시간을 출력하세요. 무엇을 관찰할 수 있나요?
