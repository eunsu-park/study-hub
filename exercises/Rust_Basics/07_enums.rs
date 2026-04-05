// Exercise: Enums and Pattern Matching
// Implement enums with match expressions.
//
// Run: rustc 07_enums.rs && ./07_enums

fn main() {
    // Exercise 1: Traffic light
    let mut light = TrafficLight::Red;
    for _ in 0..6 {
        println!("{:?}: {} seconds", light, light.duration());
        light = light.next();
    }

    // Exercise 2: Expression evaluator
    let expr = Expr::Add(
        Box::new(Expr::Num(3.0)),
        Box::new(Expr::Mul(
            Box::new(Expr::Num(4.0)),
            Box::new(Expr::Num(5.0)),
        )),
    );
    let result = eval(&expr);
    println!("3 + (4 * 5) = {result}");
    assert!((result - 23.0).abs() < f64::EPSILON);

    // Exercise 3: Safe division chain
    let result = safe_div(100, 5)
        .and_then(|r| safe_div(r, 2))
        .and_then(|r| safe_div(r, 5));
    assert_eq!(result, Some(2));

    let fail = safe_div(100, 0);
    assert_eq!(fail, None);
    println!("Division chain passed!");

    // Exercise 4: Command parser
    assert!(matches!(parse_command("add milk"), Some(Command::Add(_))));
    assert!(matches!(parse_command("remove 2"), Some(Command::Remove(2))));
    assert!(matches!(parse_command("list"), Some(Command::List)));
    assert!(matches!(parse_command("quit"), Some(Command::Quit)));
    assert!(parse_command("invalid").is_none());
    println!("Command parser passed!");

    println!("\nAll exercises passed!");
}

// Exercise 1: Traffic Light
#[derive(Debug)]
enum TrafficLight {
    Red,
    Yellow,
    Green,
}

impl TrafficLight {
    fn duration(&self) -> u32 {
        todo!()
    }
    fn next(&self) -> Self {
        todo!()
    }
}

// Exercise 2: Expression tree
enum Expr {
    Num(f64),
    Add(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
}

fn eval(expr: &Expr) -> f64 {
    todo!()
}

// Exercise 3: Safe division
fn safe_div(a: i32, b: i32) -> Option<i32> {
    todo!()
}

// Exercise 4: Command parser
#[derive(Debug)]
enum Command {
    Add(String),
    Remove(usize),
    List,
    Quit,
}

fn parse_command(input: &str) -> Option<Command> {
    // Parse "add <item>", "remove <index>", "list", "quit"
    todo!()
}
