#![allow(non_snake_case, dead_code)]

mod util;

use util::k2_graph::k2_tree::K2Tree;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn help() {
        let tree = K2Tree::test_tree();
        assert_eq!(tree.get_bit(4, 1), true);
    }
    #[test]
    fn help2() {
        let tree = K2Tree::test_tree();
        assert_eq!(tree.get_bit(4, 5), true);
    }
}