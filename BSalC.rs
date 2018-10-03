// DISCLAIMER: This code
// (a) Doesn't compile.
// (b) May never compile (ignoring syntax) without substantial changes
// to the Rust type system.
//
// To reduce visual noise, fully uppercase identifiers are uniformly treated
// as generic parameters (they are not introduced into scope explicitly unless
// they have trait constraints).
//
// e.g. fn foo<T, SU>(t: T) -> SU is shortened to fn foo(t: T) -> SU
//
// Also, this translation is not term to term, and the types don't make sense
// sometimes; don't kill me. I've tried to do my best here.
// However, if _you_ complain about not following Rust's casing/formatting
// conventions, I will certainly send you a glitter bomb.

// Fig 5. Code sample on 79:5
// "release.tar" %> \_ -> do
//     need ["release.txt"]
//     files <- lines <$> readFile "release.txt"
//     need files
//     system "tar" $ ["-cf", "release.tar"] ++ files

let releaseTarRule = buildRuleFor(
    "release.tar",
    // The argument to the lambda is ignored (using JS-style lambda syntax).
    _ => {
        need(["release.txt"]);
        let files: Vec<String> = map(lines_of, readFile("release.txt"));
        need(files);
        system("tar", ["-cf", "release.tar"] + files)
    });

// Fig 5. Code sample on 79:7
// data Store i k v -- i = info, k = key, v = value
// initialise :: i -> (k -> v) -> Store i k v
// getInfo :: Store i k v -> i
// putInfo :: i -> Store i k v -> Store i k v
// getValue :: k -> Store i k v -> v
// putValue :: Eq k => k -> v -> Store i k v -> Store i k v

// I = Info, K = Key, V = Value
struct Store<I, K, V>; // Definition not provided.

fn initialize(info : I, mapping : Fn(K) -> V) -> Store<I, K, V>;
fn getInfo(mystore : Store<I, K, V>) -> I;
// Returns an updated store because values are immutable.
fn putInfo(newinfo : I, mystore : Store<I, K, V>) -> Store<I, K, V>;
fn getValue(searchKey : K, mystore : Store<I, K, V>) -> V;
fn putValue<K: Eq>(saveKey : K, saveVal : V, mystore : Store<I, K, V>) -> Store<I, K, V>;

// Fig 5. Code sample on 79:7 (contd.)
//
// data Hash v -- a compact summary of a value with a fast equality check
// hash :: Hashable v => v -> Hash v
// getHash :: Hashable v => k -> Store i k v -> Hash v

struct Hash<V>; // Definition not provided.
fn hash<V: Hashable>(val: V) -> Hash<V>;
fn getHash<V: Hashable>(searchKey : K, mystore : Store<I, K, V>) -> Hash<V>

// Fig 5. Code sample on 79:7 (contd.)
//
// -- Build tasks (see Section 3.2)
// newtype Task c k v = Task { run :: forall f .  c f => (k -> f v) -> f v }
// type Tasks c k v = k -> Maybe ( Task c k v)
//
// (Rust actually uses just "for" but I think "forall" is clearer. The
// parameter C should be thought of as some trait constraint describing what
// kind of build system we're interested in.)
struct Task<C, K, V> = Task {
    runTask : forall<F> Fn<F: C>(Fn(K) -> F<V>) -> F<V>,
};

type Tasks<C, K, V> = Fn(K) -> Option<Task<C, K, V>>

// Fig 5. Code sample on 79:7 (contd.)
//
// -- Build system (see Section 3.3)
// type Build c i k v = Tasks c k v -> k -> Store i k v -> Store i k v
// -- Build system components: a scheduler and a rebuilder (see Section 5)
// type Scheduler c i ir k v = Rebuilder c ir k v -> Build c i k v
// type Rebuilder c   ir k v = k -> v -> Task c k v -> Task (MonadState ir) k v
type Build<C, I, K, V> = Fn(Tasks<C, K, V>, K, Store<I, K, V>) -> Store<I, K, V>;
type Scheduler<C, I, IR, K, V> = Fn(Rebuilder<C, IR, K, V>) -> Build<C, I, K, V>;
type Rebuilder<C,    IR, K, V> = Fn(K, V, Task<C, K, V>) -> Task<MonadState<IR>, K, V>;
// Pause here and explain MonadState a bit.
// TODO: Expand on this.
// What this means is that the function in the return parameter can assume that the
// F type parameter has some access to mutable state of type (parameter) IR, which
// is some Intermediate Representation of the build process.

// Fig 6. Code from 79:8
//-- Applicative functors
// pure :: Applicative f => a -> f a
// (<$>) :: Functor f => (a -> b) -> f a -> f b -- Left-associative
// (<*>) :: Applicative f => f (a -> b) -> f a -> f b -- Left-associative

trait Functor<F> {
    // Named version of (<$>)
    fn map(f: Fn(A) -> B, val: F<A>) -> F<B>;
}
trait Applicative<F> {
    fn pure(val: A) -> F<A>;
    // Notice that the function is "inside F" here, compared to map.
    // Named version of (<*>)
    fn apply(f: F<Fn(A) -> B>, val: F<A>) -> F<B>;
}

// Fig 6. Code from 79:8 (contd.)
//
// -- Standard State monad from Control.Monad.State
// data State s a
// instance Monad (State s)
// get :: State s s
// gets :: (s -> a) -> State s a
// put :: s -> State s ()
// modify :: (s -> s) -> State s ()
// runState :: State s a -> s -> (a, s)
// execState :: State s a -> s -> s
//
// -- Standard types from Data.Functor.Identity and Data.Functor.Const
// newtype Identity a = Identity { runIdentity :: a }
// newtype Const m a = Const { getConst :: m }
// instance Functor (Const m) where
//    fmap _ (Const m) = Const m
// instance Monoid m => Applicative (Const m) where
//   pure _ = Const mempty
// -- mempty is the identity of the monoid m
// Const x <*> Const y = Const (x <> y) -- <> is the binary operation of the monoid m
