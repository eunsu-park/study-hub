// 13_smart_pointers.rs — Box, Rc, RefCell, and Arc
//
// Run: rustc 13_smart_pointers.rs && ./13_smart_pointers

use std::cell::RefCell;
use std::rc::Rc;

fn main() {
    println!("=== Box<T> ===");
    box_demo();

    println!("\n=== Rc<T> ===");
    rc_demo();

    println!("\n=== RefCell<T> ===");
    refcell_demo();

    println!("\n=== Rc<RefCell<T>> ===");
    rc_refcell_demo();

    println!("\n=== Recursive Types with Box ===");
    recursive_type_demo();
}

fn box_demo() {
    // Box puts data on the heap
    let b = Box::new(5);
    println!("Box value: {b}");

    // Useful for large data you don't want on the stack
    let large = Box::new([0u8; 1_000_000]); // 1MB on heap, not stack
    println!("Large array length: {}", large.len());

    // Box as trait object
    let shapes: Vec<Box<dyn Shape>> = vec![
        Box::new(Circle { radius: 5.0 }),
        Box::new(Square { side: 3.0 }),
    ];
    for shape in &shapes {
        println!("  {} area: {:.2}", shape.name(), shape.area());
    }
}

trait Shape {
    fn area(&self) -> f64;
    fn name(&self) -> &str;
}

struct Circle {
    radius: f64,
}
impl Shape for Circle {
    fn area(&self) -> f64 {
        std::f64::consts::PI * self.radius * self.radius
    }
    fn name(&self) -> &str {
        "Circle"
    }
}

struct Square {
    side: f64,
}
impl Shape for Square {
    fn area(&self) -> f64 {
        self.side * self.side
    }
    fn name(&self) -> &str {
        "Square"
    }
}

fn rc_demo() {
    // Rc enables multiple ownership (single-threaded)
    let data = Rc::new(String::from("shared data"));
    println!("Reference count: {}", Rc::strong_count(&data));

    let clone1 = Rc::clone(&data); // Increments ref count, not deep copy
    let clone2 = Rc::clone(&data);
    println!("Reference count: {}", Rc::strong_count(&data));

    println!("clone1: {clone1}");
    println!("clone2: {clone2}");

    drop(clone1);
    println!("After drop: count = {}", Rc::strong_count(&data));
}

fn refcell_demo() {
    // RefCell enables interior mutability — borrow checking at runtime
    let data = RefCell::new(vec![1, 2, 3]);

    // Immutable borrow
    println!("Data: {:?}", data.borrow());

    // Mutable borrow
    data.borrow_mut().push(4);
    println!("After push: {:?}", data.borrow());

    // Runtime panic if rules are violated:
    // let r1 = data.borrow();
    // let r2 = data.borrow_mut(); // PANIC: already borrowed immutably
}

fn rc_refcell_demo() {
    // Rc<RefCell<T>> = shared + mutable — the "escape hatch"
    let shared_list = Rc::new(RefCell::new(Vec::<String>::new()));

    let writer1 = Rc::clone(&shared_list);
    let writer2 = Rc::clone(&shared_list);
    let reader = Rc::clone(&shared_list);

    writer1.borrow_mut().push("from writer1".to_string());
    writer2.borrow_mut().push("from writer2".to_string());

    println!("Shared list: {:?}", reader.borrow());
}

// Recursive type using Box
#[derive(Debug)]
enum List {
    Cons(i32, Box<List>),
    Nil,
}

fn recursive_type_demo() {
    // Without Box, the compiler can't determine the size of List
    // because it would be infinitely recursive
    let list = List::Cons(1, Box::new(List::Cons(2, Box::new(List::Cons(3, Box::new(List::Nil))))));

    // Print the list
    fn print_list(list: &List) {
        match list {
            List::Cons(val, next) => {
                print!("{val} -> ");
                print_list(next);
            }
            List::Nil => println!("Nil"),
        }
    }

    print_list(&list);
}
