// Exercise: Smart Pointers
// Practice with Box, Rc, RefCell.
//
// Run: rustc 13_smart_pointers.rs && ./13_smart_pointers

use std::cell::RefCell;
use std::rc::Rc;

fn main() {
    // Exercise 1: Recursive list with Box
    let list = List::Cons(1, Box::new(List::Cons(2, Box::new(List::Cons(3, Box::new(List::Nil))))));
    assert_eq!(list.sum(), 6);
    assert_eq!(list.len(), 3);
    println!("Exercise 1 passed! sum={}, len={}", list.sum(), list.len());

    // Exercise 2: Shared ownership with Rc
    let shared = Rc::new(vec![1, 2, 3]);
    let clone1 = Rc::clone(&shared);
    let clone2 = Rc::clone(&shared);
    assert_eq!(Rc::strong_count(&shared), 3);
    assert_eq!(sum_rc(&clone1), 6);
    assert_eq!(sum_rc(&clone2), 6);
    println!("Exercise 2 passed!");

    // Exercise 3: Interior mutability with Rc<RefCell<T>>
    let log = Rc::new(RefCell::new(Vec::<String>::new()));
    let writer1 = Rc::clone(&log);
    let writer2 = Rc::clone(&log);
    append_log(&writer1, "event1");
    append_log(&writer2, "event2");
    append_log(&writer1, "event3");
    assert_eq!(log.borrow().len(), 3);
    println!("Exercise 3 passed! Log: {:?}", log.borrow());

    println!("\nAll exercises passed!");
}

// Exercise 1: Implement methods on a recursive list
enum List {
    Cons(i32, Box<List>),
    Nil,
}

impl List {
    fn sum(&self) -> i32 {
        // TODO: Recursively sum all values
        todo!()
    }

    fn len(&self) -> usize {
        // TODO: Count the number of elements
        todo!()
    }
}

// Exercise 2: Function that borrows an Rc
fn sum_rc(data: &Rc<Vec<i32>>) -> i32 {
    // TODO: Sum the elements (Rc auto-derefs)
    todo!()
}

// Exercise 3: Append to a shared log
fn append_log(log: &Rc<RefCell<Vec<String>>>, entry: &str) {
    // TODO: Borrow mutably and push the entry
    todo!()
}
