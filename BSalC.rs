// Disclaimers and warnings
// ------------------------
// Obviously, this file is no substitute for the paper itself:
// https://www.microsoft.com/en-us/research/uploads/prod/2018/03/build-systems.pdf
// There is also an awesome talk (as always :D) by SPJ on YouTube:
// https://www.youtube.com/watch?v=BQVT6wiwCxM
//
// Unlike the code in the paper, the code given below
// (a) Doesn't compile.
// (b) May never compile (ignoring syntax) without substantial changes
// to the Rust type system.
//
// This translation is not term to term, and the types don't make sense
// sometimes (e.g. <$> and <*> are elided in places); don't kill me.
// I've tried to do my best here.
// However, if _you_ complain about not following Rust's casing/formatting
// conventions, I will most certainly send you a glitter bomb.
//
// Differences from ordinary Rust
// ------------------------------
// 1. To reduce visual noise, fully uppercase identifiers are uniformly treated
//    as generic parameters (they are not introduced into scope explicitly unless
//    they have trait constraints).
//
//    e.g. fn foo<T, SU>(t : T) -> SU is shortened to fn foo(t : T) -> SU
//
// 2. I'm using JS-style lambda syntax (everywhere) + explicit returns (randomly).
//    e.g. (foo, bar) => { baz(foo); return qux(foo, bar); }
//
//    Such a function will have type Fn(Foo, Bar) -> Qux
//
// 3. Type parameters are curried Foo<T, S> = Foo<T><S>.
//    e.g. we can say that Vec (not Vec<T>) is a "mappable container".
//
// 4. Record definition uses an `=` sign instead of `:`
//    e.g. { x = 10, y = 20 } instead of { x : 10, y : 10 }

//------------------------------------------------------------------------------

// Fig 5. Code on 79:5
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
        let files: Vec<String> = readFile("release.txt").splitLines();
        need(files);
        return system("tar", ["-cf", "release.tar"] + files);
    });

// Fig 5. Code on 79:7
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

// Fig 5. Code on 79:7 (contd.)
//
// data Hash v -- a compact summary of a value with a fast equality check
// hash :: Hashable v => v -> Hash v
// getHash :: Hashable v => k -> Store i k v -> Hash v

struct Hash<V>; // Definition not provided.
fn hash<V: Hashable>(val : V) -> Hash<V>;
fn getHash<V: Hashable>(searchKey : K, mystore : Store<I, K, V>) -> Hash<V>

// Fig 5. Code on 79:7 (contd.)
//
// -- Build tasks (see Section 3.2)
// newtype Task c k v = Task { run :: forall f .  c f => (k -> f v) -> f v }
// type Tasks c k v = k -> Maybe ( Task c k v)
//
// --------
// (Rust actually uses just "for" but I think "forall" is clearer.)
//
// C = Constraint, (and K = Key, V = Value as earlier)
//
// The parameter C should be thought of as some trait constraint describing what
// kind of build system we're interested in.
//
// Since runTask works for all choices of F, we can pick F (by supplying the
// argument to runTask which has type Fn(K) -> F<V>) depending on what kind
// of build output we want. For example, there are choices for F which can:
// 1. Be used to just list dependencies (no actual build).
// 2. Be used to actually build things (doing IO).
//
// See Section 3.2-3.4 in the paper for more details on C and F.
struct Task<C, K, V> = Task {
    runTask : forall<F> Fn<F: C>(Fn(K) -> F<V>) -> F<V>,
};

// Input keys are associated with None (because there is nothing to build,
// the thing is given as input), and non-input keys (for things that need to be
// computed based off input keys) are associated with Some(task).
// (Section 3.2)
type Tasks<C, K, V> = Fn(K) -> Option<Task<C, K, V>>

// Fig 5. Code on 79:7 (contd.)
//
// -- Build system (see Section 3.3)
// type Build c i k v = Tasks c k v -> k -> Store i k v -> Store i k v
// -- Build system components: a scheduler and a rebuilder (see Section 5)
// type Scheduler c i ir k v = Rebuilder c ir k v -> Build c i k v
// type Rebuilder c   ir k v = k -> v -> Task c k v -> Task (MonadState ir) k v

type Build<C, I, K, V> = Fn(Tasks<C, K, V>, K, Store<I, K, V>) -> Store<I, K, V>;

type Scheduler<C, I, IR, K, V> = Fn(Rebuilder<C, IR, K, V>) -> Build<C, I, K, V>;
type Rebuilder<C,    IR, K, V> = Fn(K, V, Task<C, K, V>) -> Task<MonadState<IR>, K, V>;

// MonadState is explained a bit later in this document.
//

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

// May be loosely thought of as a pair of values, one of type S ("mutable" state)
// and A (type of whatever value is captured). You might be wondering why we
// have all these small functions to do simple things instead of "just"
// working with a tuple. Well, the answer is that the Haskell implementation
// of State is _not_ actually a pair, and these functions define an interface
// through which we can interact with values of type A wrapped inside State<S, A>
struct State<S, A>; // Definition skipped.
fn get() -> State<S, S>;
fn gets(f: Fn(S) -> A) -> State<S, A>;
fn put(astate: S) -> State<S, ()>;
fn modify(statechange: Fn(S) -> S) -> State<S, ()>;
fn  runState(val: State<S, A>, startState: S) -> (A, S)
fn execState(val: State<S, A>, startState: S) ->     S

// -- Standard types from Data.Functor.Identity and Data.Functor.Const
// newtype Identity a = Identity { runIdentity :: a }
// newtype Const m a = Const { getConst :: m }
// instance Functor (Const m) where
//    fmap _ (Const m) = Const m
// instance Monoid m => Applicative (Const m) where
//   pure _ = Const mempty
// -- mempty is the identity of the monoid m
// Const x <*> Const y = Const (x <> y) -- <> is the binary operation of the monoid m

// This one may seem weird but it is needed as can't make a Functor out of
// "nothing". Whereas now that we have a singleton container, Identity can
// implement the Functor/Applicative/Monad traits (definitions elided).
struct Identity<A> { runIdentity : A }

struct Const<M, A> { getConst : M }
// This means that we can "map over" the A type parameter because Const<M, A> = Const<M><A>.
impl Functor<Const<M>> {
    // The function is ignored, we don't have anything to apply it to.
    fn map(_: Fn(A) -> B, val: Const<M><A>) -> Const<M><B> {
        return { getConst = val.getConst };
    }
}
impl Applicative<Const<M: Monoid>> {
    fn pure(_: A) -> Const<M, A> {
        return { getConst = M::MONOID_ZERO };
    }
    fn apply(f : Const<M, Fn(A) -> B>, val : Const<M, A>) -> Const<M, B> {
        return { getConst = M::monoid_combine(f.getConst, val.getConst) };
    }
}

// Repeated definitions from earlier.

struct Task<C, K, V> = Task {
    runTask : forall<F> Fn<F: C>(Fn(K) -> F<V>) -> F<V>,
};
type Tasks<C, K, V> = Fn(K) -> Option<Task<C, K, V>>

// Spreadsheet example
//  A1: 10  B1: A1 + A2
//  A2: 20  B2: B1 * 2
// sprsh1   :: Tasks Applicative String Integer
// sprsh1   "B1" = Just $ Task $ \fetch -> ((+) <$> fetch "A1" <*> fetch "A2")
// sprsh1   "B2" = Just $ Task $ \fetch -> ((*2) <$> fetch "B1")
// sprsh1   _ = Nothing
let spreadsheet1 : Tasks<Applicative, String, Integer> =
    cellname => {
        match cellname {
            "B1" => Some(Task(fetch => fetch("A1") + fetch("A2"))),
            "B2" => Some(Task(fetch => fetch("B1") * 2)),
            _    => None,
        }
    };

// Code on 79:9
type Build<C, I, K, V> = Fn(Tasks<C, K, V>, K, Store<I, K, V>) -> Store<I, K, V>;

// Code on 79:9 (contd.)
//
// From the paper -
// The busy build system defines the callback `fetch` so that, when given a key,
// it brings the key up to date in the store, and returns its value. The
// function fetch runs in the standard Haskell State monad - see Fig. 6 -
// initialised with the incoming store by execState. To bring a key up to date,
// `fetch` asks the task description tasks how to compute the value of k.
// If tasks returns Nothing the key is an input, so `fetch` simply reads the
// result from the store. Otherwise `fetch` runs the obtained task to produce
// a resulting value v, records the new key/value mapping in the store, and
// returns v. Notice that `fetch` passes itself to task as an argument,
// so that the latter can use fetch to recursively find the values of k’s
// dependencies.
//
// busy :: Eq k => Build Applicative () k v
// busy tasks key store = execState (fetch key) store
//   where
//     fetch :: k -> State (Store () k v) v
//     fetch k = case tasks k of
//       Nothing -> gets (getValue k)
//       Just task -> do v <- run task fetch; modify (putValue k v); return v
let fetch : Fn(K) -> State<Store<(), K, V>, V> =
    k => match tasks k {
        None => {
            let valGetter: Fn(Store<(), K, V>) -> V = (store => getValue(k, store));
            return gets(valGetter);
        }
        Some(task) -> {
            let v: V = run(task, fetch); // <-- Recursive call
            modify(putValue(k, v));
            return v;
        }
    };
let busy : Build<Applicative, (), K: Eq, V> =
    (tasks, key, store) => execState(fetch(key), store)

// Spreadsheet example (same as earlier)
//  A1: 10  B1: A1 + A2
//  A2: 20  B2: B1 * 2
//
// REPL output on 79:9
// λ> store = initialise () (\key -> if key == "A1" then 10 else 20)
// λ> result = busy sprsh1 "B2" store
// λ> getValue "B1" result
// 30
// λ> getValue "B2" result
// 60
// >>> store = initialise((), key => if key == "A1" then 10 else 20)
// >>> result = busy(spreadsheet1, "B2", store)
// >>> getValue("B1", result)
// 30
// >>> getValue("B2", result)
// 60
