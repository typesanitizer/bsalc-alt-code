// Disclaimers and warnings
// ========================
// Obviously, this file is no substitute for the paper itself:
// https://www.microsoft.com/en-us/research/uploads/prod/2018/03/build-systems.pdf
// There is also an awesome talk (as always :D) by SPJ on YouTube:
// https://www.youtube.com/watch?v=BQVT6wiwCxM
//
// Unlike the code in the paper, the code given below
// (a) Doesn't compile.
// (b) May never compile (ignoring syntax) without substantial changes
//     to the Rust type system.[FN1]
//
// Stuff marked with {Comment} is additional commentary provided by me.
// It is primarily written for stuff that may seem weird for non-Haskell folks.
// Don't worry though, there is no monad tutorial here. :)
//
// If you have trouble understanding something and/or concrete suggestions to
// improve the translation, please open an issue on the GitHub issue tracker!
//
// [FN1]: I'm not complaining here, please do not misinterpret it as such.
//
// For Haskellers:
// ---------------
// This translation is not term to term, and the types don't make sense
// sometimes (e.g. <$>, <*>, >>= and lifts are elided in places); don't kill me.
// I've tried to do my best here. Also, you're surely better off reading the
// paper directly; all the extra brackets here may make your head hurt.
//
// For Rustaceans:
// ---------------
// If you complain about not following Rust's casing/formatting conventions,
// I will almost surely send you a glitter bomb.
//
// Differences from ordinary Rust
// ==============================
// 1. To reduce visual noise, fully uppercase identifiers are uniformly treated
//    as generic parameters (they are not introduced into scope explicitly unless
//    they have trait constraints).
//
//    e.g. fn foo<T, SU>(t: T) -> SU is shortened to fn foo(t: T) -> SU
//
// 2. I'm using JS-style lambda syntax (everywhere) + explicit returns (randomly).
//    e.g. (foo, bar) => { baz(foo); return qux(foo, bar); }
//
//    Such a function will have type Fn(Foo, Bar) -> Qux
//
// 3. Type parameters are curried Foo<T, S> = Foo<T><S>.
//    e.g. we can say that List (not List<T>) is a "mappable container".


// Before we begin, let's pause for a haiku -
//
// In this world
// We walk on the roof of hell
// Gazing at flowers
//
// - Kobayashi Issa
//
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
        let files: List<String> = readFile("release.txt").splitLines();
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

fn initialize(info: I, mapping: Fn(K) -> V) -> Store<I, K, V>;
fn getInfo(mystore: Store<I, K, V>) -> I;
// Returns an updated store because values are immutable.
fn putInfo(newinfo: I, mystore: Store<I, K, V>) -> Store<I, K, V>;
fn getValue(searchKey: K, mystore: Store<I, K, V>) -> V;
fn putValue<K: Eq>(saveKey: K, saveVal: V, mystore: Store<I, K, V>) -> Store<I, K, V>;

// Fig 5. Code on 79:7 (contd.)
//
// data Hash v -- a compact summary of a value with a fast equality check
// hash :: Hashable v => v -> Hash v
// getHash :: Hashable v => k -> Store i k v -> Hash v

struct Hash<V>; // Definition not provided.
fn hash<V: Hashable>(val: V) -> Hash<V>;
fn getHash<V: Hashable>(searchKey: K, mystore: Store<I, K, V>) -> Hash<V>

// Fig 5. Code on 79:7 (contd.)
//
// -- Build tasks (see Section 3.2)
// newtype Task c k v = Task { run :: forall f .  c f => (k -> f v) -> f v }
// type Tasks c k v = k -> Maybe ( Task c k v)
//
// --------
// {Comment} (Rust actually uses just "for" but I think "forall" is clearer.)
//
// C = Constraint, (and K = Key, V = Value as earlier)
//
// The parameter C should be thought of as some trait constraint describing what
// kind of build system we're interested in.
//
// Since `run` works for all choices of F, we can pick F (by supplying the
// argument to `run` which has type Fn(K) -> F<V>) depending on what kind
// of build output we want. For example, there are choices for F which can:
// 1. Be used to just list dependencies (no actual build).
// 2. Be used to actually build things (doing IO).
//
// See Section 3.2-3.4 in the paper for more details on C and F.
struct Task<C, K, V> {
    run: forall<F> Fn<F: C>(Fn(K) -> F<V>) -> F<V>,
};

// Input keys are associated with None (because there is nothing to build,
// the thing is given as input), and non-input keys (for things that need to be
// computed based off input keys) are associated with Some(task).
// (Section 3.2)
type Tasks<C, K, V> = Fn(K) -> Option<Task<C, K, V>>;

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

// {Comment} MonadState is explained a bit later in this document.

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
    // Notice that the function (first argument) is "inside F" here, unlike map.
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

// {Comment} May be loosely thought of as a pair of values, one of type S ("mutable"
// state) and another of type A (type of whatever value is captured).
// You might be wondering why we have all these small functions to do simple
// things instead of "just" working with a tuple. Well, the answer is that the
// State is _not_ actually a pair (otherwise, runState and execState wouldn't make
// much sense), but a function of type Fn(S) -> (S, A).
struct State<S, A>; // Definition skipped.
fn get() -> State<S, S>;
fn gets(f: Fn(S) -> A) -> State<S, A>;
fn put(astate: S) -> State<S, ()>;
fn modify(statechange: Fn(S) -> S) -> State<S, ()>;
fn  runState(val: State<S, A>, startState: S) -> (A, S);
fn execState(val: State<S, A>, startState: S) ->     S;

// -- Standard types from Data.Functor.Identity and Data.Functor.Const
// newtype Identity a = Identity { runIdentity :: a }
// newtype Const m a = Const { getConst :: m }
// instance Functor (Const m) where
//    fmap _ (Const m) = Const m
// instance Monoid m => Applicative (Const m) where
//   pure _ = Const mempty
// -- mempty is the identity of the monoid m
// Const x <*> Const y = Const (x <> y) -- <> is the binary operation of the monoid m

// {Comment} This one may seem weird but it is needed as can't make a Functor out of
// "nothing". Whereas now that we have a singleton container, Identity can
// implement the Functor/Applicative/Monad traits (definitions elided).
struct Identity<A> { runIdentity: A }

struct Const<M, A> { getConst: M }
// This means that we can "map over" the A type parameter because Const<M, A> = Const<M><A>.
impl Functor<Const<M>> {
    // The function is ignored, we don't have anything to apply it to.
    fn map(_: Fn(A) -> B, val: Const<M><A>) -> Const<M><B> {
        return Const { getConst: val.getConst };
    }
}
impl Applicative<Const<M: Monoid>> {
    fn pure(_: A) -> Const<M, A> {
        return Const { getConst: M::MONOID_ZERO };
    }
    fn apply(f: Const<M, Fn(A) -> B>, val: Const<M, A>) -> Const<M, B> {
        return Const { getConst: M::monoid_combine(f.getConst, val.getConst) };
    }
}

// Repeated definitions from earlier.

struct Task<C, K, V> {
    run: forall<F> Fn<F: C>(Fn(K) -> F<V>) -> F<V>,
};
type Tasks<C, K, V> = Fn(K) -> Option<Task<C, K, V>>;

// Spreadsheet example
//  A1: 10  B1: A1 + A2
//  A2: 20  B2: B1 * 2
// sprsh1 :: Tasks Applicative String Integer
// sprsh1 "B1" = Just $ Task $ \fetch -> ((+) <$> fetch "A1" <*> fetch "A2")
// sprsh1 "B2" = Just $ Task $ \fetch -> ((*2) <$> fetch "B1")
// sprsh1 _    = Nothing
let spreadsheet1: Tasks<Applicative, String, Integer> =
    cellname => {
        match cellname {
            "B1" => Some(Task { run: fetch => fetch("A1") + fetch("A2") }),
            "B2" => Some(Task { run: fetch => fetch("B1") * 2 }),
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
let busy: Build<Applicative, (), K: Eq, V> =
    (tasks, key, store) => {
        let fetch: Fn(K) -> State<Store<(), K, V>, V> =
            k => match tasks(k) {
                None => {
                    let valGetter: Fn(Store<(), K, V>) -> V =
                        store => getValue(k, store);
                    return gets(valGetter);
                },
                Some(task) => {
                    let v: V = task.run(fetch); // <-- Recursive call
                    modify(store => putValue(k, v, store));
                    return v;
                }
            };
        return execState(fetch(key), store);
    }
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
//
>>> store = initialise((), key => if key == "A1" then 10 else 20)
>>> result = busy(spreadsheet1, "B2", store)
>>> getValue("B1", result)
30
>>> getValue("B2", result)
60

// Spreadsheet example (new)
//
// A1: 10   B1: IF(C1=1,B2,A2)   C1:1
// A2: 20   B2: IF(C1=1,A1,B1)
//
// Code on 79:10
// sprsh2 :: Tasks Monad String Integer
// sprsh2 "B1" = Just $ Task $ \fetch -> do
//     c1 <- fetch "C1"
//     if c1 == 1 then fetch "B2" else fetch "A2"
// sprsh2 "B2" = Just $ Task $ \fetch -> do
//     c2 <- fetch "C1"
//     if c1 == 1 then fetch "A1" else fetch "B1"
// sprsh2 _ = Nothing
let spreadsheet2: Tasks<Monad, String, Integer> =
    cellname => match cellname {
        "B1" => Some(Task {
            run: fetch => {
                let c1 = fetch("C1");
                if c1 == 1 { return fetch("B2"); } else { return fetch("A2"); }
            }
        }),
        "B2" => Some(Task {
            run: fetch => {
                let c1 = fetch("C2");
                if c1 == 1 { return fetch("A1"); } else { return fetch("B1"); }
            }
        }),
        _ => Nothing
    };

// Code on 79:11
// compute :: Task Monad k v -> Store i k v -> v
// compute task store = runIdentity $ run task (\k -> Identity (getValue k store))
fn compute(task: Task<Monad, K, V>, store: Store<I, K, V>) -> V {
    // Recall task is a struct with a single field "run".
    // Identity is a struct with a single field "runIdentity".
    return task.run(k => Identity { runIdentity: getValue(k, store) }).runIdentity;
}

// Code on 79:11
//
// Definition 3.1 (Correctness)
//
// build :: Build c i k v
// tasks :: Tasks c k v
// key :: k
// store, result :: Store i k v
// result = build tasks key store
//
// The build result is correct if the following two conditions hold:
// • result and store agree on inputs, that is, for all input keys k ∈ I :
//     getValue k result == getValue k store.
//   In other words, no inputs were corrupted during the build.
// • The result is consistent with the tasks, i.e. for all non-input keys k ∈ O, the result of
//   recomputing the corresponding task matches the value stored in the result:
//     getValue k result == compute task result.

let build: Build<C, I, K, V>;
let tasks: Tasks<C, K, V>;
let key: K;
let store: Store<I, K, V>;
let result: Store<I, K, V> = build(tasks, key, store);

      getValue(k, result) == getValue(k, store)
      getValue(k, result) == compute(task, result)

// Code on 79:12
//
// dependencies :: Task Applicative k v -> [k]
// dependencies task = getConst $ run task (\k -> Const [k])
//
// λ> dependencies $ fromJust $ sprsh1 "B1"
// ["A1","A2"]
// λ> dependencies $ fromJust $ sprsh1 "B2"
// ["B1"]
fn dependencies(task: Task<Applicative, K, V>) -> List<K> {
    return task.run(k => Const { getConst: list![k] }).getConst;
}
// {Comment} unwrap() asserts that the value matches Some(x) and extracts x
>>> dependencies(spreadsheet1("B1").unwrap())
["A1", "A2"]
>>> dependencies(spreadsheet1("B2").unwrap())
["B1"]

// Code on 79:12
// import Control.Monad.Writer
// track :: Monad m => Task Monad k v -> (k -> m v) -> m (v, [(k, v)])
// track task fetch = runWriterT $ run task trackingFetch
//   where
//     trackingFetch :: k -> WriterT [(k, v)] m v
//     trackingFetch k = do v <- lift (fetch k); tell [(k, v)]; return v
//
use control::monad::writer::*;
fn track<M: Monad>(task: Task<Monad, K, V>, fetch: Fn(K) -> M<V>) -> M<(V, List<(K, V)>)> {
    // {Comment} "Writer" may be thought of as a logging mechanism, where "tell" records
    // something in the log.
    let trackingFetch: Fn(K) -> WriterT<List<(K, V)>, M, V> =
        k => {
            let v = fetch(k);
            tell(list![(k, v)]);
            return v;
        };
    let taskoutput: WriterT<List<(K, V)> M, V> = task.run(trackingFetch);
    // {Comment} WriterT is a struct with a single field called runWriterT.
    return taskoutput.runWriterT;
}

// REPL output on 79:13
// λ> fetchIO k = do putStr (k ++ ": "); read <$> getLine
// λ> track (fromJust $ sprsh2 "B1") fetchIO
// C1: 1
// B2: 10
// (10,[("C1",1),("B2",10)])
// λ> track (fromJust $ sprsh2 "B1") fetchIO
// C1: 2
// A2: 20
// (20,[("C1",2),("A2",20)])
>>> let fetchIO = (k => {print(k + ": "); return readLine();});
>>> track(spreadsheet2("B1").unwrap(), fetchIO)
C1: 1
B2: 10
(10,[("C1",1),("B2",10)])
>>> track(spreadsheet2("B2").unwrap(), fetchIO)
C1: 2
A2: 20
(20,[("C1",2),("A2",20)])

// Code on 79:14
//
// recordVT :: k -> Hash v -> [(k, Hash v)] -> VT k v -> VT k v
// verifyVT :: (Monad m, Eq k, Eq v) => k -> Hash v -> (k -> m (Hash v)) -> VT k v -> m Bool
//
// {Comment} Maybe the type signatures have been written all in 1 line due
// to space restrictions in the paper; people usually don't format long
// signatures like this.
fn recordVT(key: K, hash: Hash<V>, trace: List<(K, Hash<V>)>, store: VT<K, V>)
            -> VT<K, V>;

fn verifyVT(key: K, hash: Hash<V>, fetch: Fn(K) -> M<V>) -> M<Bool>
  where M: Monad, K: Eq, V: Eq;

// Code on 79:15
//
// recordCT :: k -> v -> [(k, Hash v)] -> CT k v -> CT k v
// constructCT :: (Monad m, Eq k, Eq v) => k -> (k -> m (Hash v)) -> CT k v -> m [v]

fn recordCT(key: K, val: V, trace: List<(K, Hash<V>)>, store: CT<K, V>) -> CT<K, V>;

fn constructCT(key: K, fetch_hash: Fn(K) -> M<Hash<V>>, store: CT<K, V>) -> M<List<V>>
    where M: Monad, K: Eq, V: Eq;

// Code on 79:15
//
// data Trace k v r = Trace { key :: k, depends :: [(k, Hash v)], result :: r }
//
struct Trace<K, V, R> {
    key: K,
    depends: List<(K, Hash<V>)>,
    result: R,
}

// Code on 79:16
// type Scheduler c i ir k v = Rebuilder c ir k v -> Build c i k v
// type Rebuilder c   ir k v = k -> v -> Task c k v -> Task (MonadState ir) k v
//
// Repeated from earlier

type Scheduler<C, I, IR, K, V> = Fn(Rebuilder<C, IR, K, V>) -> Build<C, I, K, V>;
type Rebuilder<C,    IR, K, V> = Fn(K, V, Task<C, K, V>) -> Task<MonadState<IR>, K, V>;

// Fig 7. Code on 79:17
//
// -- Make build system; stores current time and file modification times
// type Time = Integer
// type MakeInfo k = (Time, Map k Time)
//
// make :: Ord k => Build Applicative (MakeInfo k) k v
// make = topological modTimeRebuilder
//
// -- A task rebuilder based on file modification times
// modTimeRebuilder :: Ord k => Rebuilder Applicative (MakeInfo k) k v
// modTimeRebuilder key value task = Task $ \fetch -> do
//   (now, modTimes) <- get
//   let dirty = case Map.lookup key modTimes of
//           Nothing -> True
//           time -> any (\d -> Map.lookup d modTimes > time) (dependencies task)
//   if not dirty then return value else do
//     put (now + 1, Map.insert key now modTimes)
//     run task fetch

type Time = Integer;
type MakeInfo<K> = (Time, Map<K, Time>);

let make: Build<Applicative, MakeInfo<K>, K: Ord, V> =
    (tasks, key, store) => {
        return topological(modTimeRebuilder, tasks, key, store);
    };

let modTimeRebuilder: Rebuilder<Applicative, MakeInfo<K>, K: Ord, V> =
    (key, value, task) => Task {
        run: fetch => {
            let (now: Time, modTimes: Map<K, Time>) = get();
            let dirty = match modTimes.lookup(key) {
                None => True,
                time => dependencies(task).any(dep => modTimes.lookup(dep) > time)
            }
            if dirty {
                put((now + 1, modTimes.insert(key, now))); // Save updated state.
                return task.run(fetch);
            } else {
                return value;
            }
        }
    };

// Fig 7. Code on 79:17 (contd.)
//
//   -- A topological task scheduler
// topological :: Ord k => Scheduler Applicative i i k v
// topological rebuilder tasks target = execState $ mapM_ build order
//   where
//     build :: k -> State (Store i k v) ()
//     build key = case tasks key of
//       Nothing -> return ()
//       Just task -> do
//         store <- get
//         let value = getValue key store
//             newTask :: Task (MonadState i) k v
//             newTask = rebuilder key value task
//             fetch :: k -> State i v
//             fetch k = return (getValue k store)
//         newValue <- liftStore (run newTask fetch)
//         modify $ putValue key newValue
//     order = topSort (reachable dep target)
//     dep k = case tasks k of { Nothing -> []; Just task -> dependencies task }
//
// {Comment} As Rust doesn't have currying-by-default, I've tried to make the
// signatures match up with the Rust-y signatures, so the implementation looks
// a bit weird.
let topological: Scheduler<Applicative, I, I, K: Ord, V> =
    rebuilder => {
        return (tasks, target: K, startStore) => {
            let build: Fn(K) -> State<Store<I, K, V>, ()> =
                key => match tasks(key) {
                    None => { return (); },
                    Some(task) => {
                        store = get();
                        let value = getValue(key, store);
                        let newTask: Task<MonadState<I>, K, V> = rebuilder(key, value, task);
                        let fetch: Fn(K) -> State<I, V> =
                            k => { return getValue(k, store); };
                    }
                    let newValue = liftStore(newTask.run(fetch));
                    modify(store => putValue(key, newValue, store));
                };
            let order: List<K> = topSort(reachable(depsOf, target));
            let depsOf: Fn(K) -> List<K> =
                key => match tasks(key) {
                    None => List::Nil,
                    Some(task) => dependencies(task),
                };
            // {Comment} mapM is similar to map except that applying map will give
            // List<State<S, A>> whereas applying mapM will give State<S, List<A>> by
            // threading the state sequentially, which can then be executed with an
            // initial store to give a final store.
            return execState(order.mapM(build), startStore);
        };
    };

// Fig 7. Code on 79:17 (contd.)
//
// -- Standard graph algorithms (implementation omitted)
// reachable :: Ord k => (k -> [k]) -> k -> Graph k
// topSort :: Ord k => Graph k -> [k] -- Throws error on a cyclic graph
//
// -- Expand the scope of visibility of a stateful computation
// liftStore :: State i a -> State (Store i k v) a
// liftStore x = do
//   (a, newInfo) <- gets (runState x . getInfo)
//   modify (putInfo newInfo)
//   return a
fn reachable<K: Ord>(depsOf: Fn(K) -> List<K>, root: K) -> Graph<K>;
fn topSort<K: Ord>(depGraph: Graph<K>) -> List<K>;

fn liftStore(x: State<I, A>) -> State<Store<I, K, V>, A> {
    let (a, newInfo) = gets(state => runState(x, getInfo(state)));
    modify(state => putInfo(newInfo, state));
    return a;
}

// Fig 8. Code on 79:19
//
// -- Excel build system; stores a dirty bit per key and calc chain
// type Chain k = [k]
// type ExcelInfo k = (k -> Bool, Chain k)
//
// excel :: Ord k => Build Monad (ExcelInfo k) k v
// excel = restarting dirtyBitRebuilder
//
// -- A task rebuilder based on dirty bits
// dirtyBitRebuilder :: Rebuilder Monad (k -> Bool) k v
// dirtyBitRebuilder key value task = Task $ \fetch -> do
//     isDirty <- get
//     if isDirty key then run task fetch else return value

type Chain<K> = List<K>;
type ExcelInfo<K> = (Fn(K) -> Bool, Chain<K>);

let excel: Build<Monad, ExcelInfo<K>, K, V> where K: Ord =
    restarting(dirtyBitRebuilder);

let dirtyBitRebuilder: Rebuilder<Monad, Fn(K) -> Bool, K, V> =
    (key, value, task) => Task {
        run: fetch => {
            let isDirty = get();
            if isDirty(key) { return task.run(fetch); } else { return value; }
        }
    };

// Fig 8. Code on 79:19 (contd.)
//
// -- A restarting task scheduler
// restarting :: Ord k => Scheduler Monad (ir, Chain k) ir k v
// restarting rebuilder tasks target = execState $ do
//     chain <- gets (snd . getInfo)
//     newChain <- liftChain $ go Set.empty $ chain ++ [target | target `notElem` chain]
//     modify $ mapInfo $ \(ir, _) -> (ir, newChain)
//   where
//     go :: Set k -> Chain k -> State (Store ir k v) (Chain k)
//     go _    []         = return []
//     go done (key:keys) = case tasks key of
//       Nothing -> (key :) <$> go (Set.insert key done) keys
//       Just task -> do
//         store <- get
//         let newTask :: Task (MonadState ir) k (Either k v)
//             newTask = try $ rebuilder key (getValue key store) task
//             fetch :: k -> State ir (Either k v)
//             fetch k | k `Set.member` done = return $ Right (getValue k store)
//                     | otherwise = return $ Left k
//         result <- liftStore (run newTask fetch) -- liftStore is defined in Fig. 7
//         case result of
//           Left dep -> go done $ dep: filter (/= dep) keys ++ [key]
//           Right newValue -> do modify $ putValue key newValue
//                                (key :) <$> go (Set.insert key done) keys

let restarting: Scheduler<Monad, (IR, Chain<K>), IR, K, V> =
    (rebuilder, tasks, target) => {
        let go: Fn(Set<K>, Chain<K>) -> State<Store<IR, K, V>, Chain<K>> =
            (done, allKeys) => match allKeys {
                List::Nil => { return List::Nil; }
                List::Cons(key, keys) => match tasks(key) {
                    None => { return list![key] + go(set::insert(key, done), keys); },
                    Some(task) => {
                        let store = get();
                        let newTask: Task<MonadState<IR>, K, Result<K, V>> =
                            try(rebuilder(key, getValue(key, store), task));
                        let fetch: Fn(K) -> State<IR, Result<K, V>> =
                            k => if done.contains(k) {
                                return Ok(getValue(k, store));
                            } else {
                                return Err(k);
                            }
                        let result = liftStore(newTask.run(fetch));
                        match result {
                            Err(dep) => {
                                // {Comment} Build the unbuilt dependency first
                                // by putting it at the front of the list.
                                let updatedKeys =
                                    list![dep] + keys.filter(k => k != dep) + list![key];
                                return go(done, updatedKeys);
                            },
                            Ok(newValue) => {
                                modify(store => putValue(key, newValue, store));
                                return list![key] + go(set::insert(key, done), keys);
                            }
                        }
                    }
                }
            };
        let mut chain = gets(state => getInfo(state).second);
        if chain.doesNotContain(target) { chain.append(target); }
        let newChain = go(set::empty, chain);
        modify(state => mapInfo((ir, _) => (ir, newChain)));
    };

// Fig 8. Code on 79:19 (contd.)
//
// -- Convert a total task into a task that accepts a partial fetch callback
// try :: Task (MonadState i) k v -> Task (MonadState i) k (Either e v)
// try task = Task $ \fetch -> runExceptT $ run task (ExceptT . fetch)
//
// -- Expand the scope of visibility of a stateful computation (implementation omitted)
// liftChain :: State (Store ir k v) a -> State (Store (ir, Chain [k]) k v) a
fn try(task: Task<MonadState<I>, K, V>) -> Task<MonadState<I>, K, Result<E, V>> {
    return Task { run: fetch => runExceptT(task.run(k => ExceptT(fetch(k)))) };
}

fn liftChain(action: State<Store<IR, K, V>, A>) ->
    State<Store<(IR, Chain<List<K>>), K, V>, A>;

// Fig 9. Code on 79:20
//
// -- Shake build system; stores verifying traces
// shake :: (Ord k, Hashable v) => Build Monad (VT k v) k v
// shake = suspending vtRebuilder
//
// -- A task rebuilder based on verifying traces
// vtRebuilder :: (Eq k, Hashable v) => Rebuilder Monad (VT k v) k v
// vtRebuilder key value task = Task $ \fetch -> do
//     upToDate <- verifyVT key (hash value) (fmap hash . fetch) =<< get
//     if upToDate then return value else do
//         (newValue, deps) <- track task fetch
//         modify $ recordVT key (hash newValue) [ (k, hash v) | (k, v) <- deps ]
//         return newValue
let shake: Build<Monad, VT<K, V>, K: Ord, V: Hashable> = suspending(vtRebuilder);

let vtRebuilder: Rebuilder<Monad, VT<K, V>, K: Eq, V: Hashable> =
    (key, value, task) => Task {
        run: fetch => {
            let upToDate = verifyVT(key, hash(value), k => fetch(k).map(hash), get());
            if upToDate {
                return value;
            } else {
                let (newValue, deps) = track(task, fetch);
                modify(state => recordVT(
                    key,
                    hash(newValue),
                    deps.map((k, v) => (k, hash(v))),
                    state)
                );
                return newValue;
            }
        }
    };

// Fig 9. Code on 79:20 (contd.)
// -- A suspending task scheduler
// suspending :: Ord k => Scheduler Monad i i k v
// suspending rebuilder tasks target store = fst $ execState (fetch target) (store, Set.empty)
//   where
//     fetch :: k -> State (Store i k v, Set k) v
//     fetch key = do
//         done <- gets snd
//         case tasks key of
//             Just task | key `Set.notMember` done -> do
//                 value <- gets (getValue key . fst)
//                 let newTask :: Task (MonadState i) k v
//                     newTask = rebuilder key value task
//                 newValue <- liftRun newTask fetch
//                 modify $ \(s, d) -> (putValue key newValue s, Set.insert key d)
//                 return newValue
//             _ -> gets (getValue key . fst) -- fetch the existing value

let suspending: Scheduler<Monad, I, I, K: Ord, V> =
    (rebuilder, tasks, target, store) => {
        let fetch: Fn(K) -> State<(Store<I, K, V>, Set<K>), V> =
            key => {
                // s.first gets the store, s.second gets the keys we've finished processing
                let done = gets(s => s.second);
                match tasks(key) {
                    Some(task) if done.doesNotContain(key) => {
                        let value = gets(s => getValue(key, s.first));
                        let newTask: Task<MonadState<I>, K, V>
                            = rebuilder(key, value, task);
                        let newValue = liftRun(newTask, fetch);
                        modify((s, d) => (putValue(key, newValue, s), Set.insert key d));
                        return newValue;
                    }
                    _ => { return gets(s => getValue(key, s.first)); }
                }
            };
        return execState(fetch(target), (store, set::empty)).first;
    };

// Fig 9. Code on 79:20 (contd.)
//
// -- Run a task using a callback that operates on a larger state (implementation omitted)
// liftRun
//   :: Task (MonadState i) k v
//   -> (k -> State (Store i k v, Set k) v)
//   -> State (Store i k v, Set k) v
fn liftRun(
    task: Task<MonadState<I>, K, V>,
    fetch: Fn(K) -> State<(Store<I, K, V>, Set<K>), V>
  ) -> State<(Store<I, K, V>, Set<K>), V>;

// Fig 10. Code on 79:22
//
// -- Bazel build system; stores constructive traces
// bazel :: (Ord k, Hashable v) => Build Monad (CT k v) k v
// bazel = restarting2 ctRebuilder -- implementation of ’restarting2’ is omitted (22 lines)
//
// -- A rebuilder based on constructive traces
// ctRebuilder :: (Eq k, Hashable v) => Rebuilder Monad (CT k v) k v
// ctRebuilder key value task = Task $ \fetch -> do
//     cachedValues <- constructCT key (fmap hash . fetch) =<< get
//     case cachedValues of
//         _ | value `elem` cachedValues -> return value
//         cachedValue:_ -> return cachedValue
//         [] -> do (newValue, deps) <- track task fetch
//                  modify $ recordCT key newValue [ (k, hash v) | (k, v) <- deps ]
//                  return newValue

let bazel: Build<Monad, CT<K, V>, K: Ord, V: Hashable> =
    restarting2(ctRebuilder);

let ctRebuilder: Rebuilder<Monad, CT<K, V>, K: Ord, V: Hashable> =
    (key, value, task) => Task {
        run: fetch => {
            let cachedValues = constructCT(key, k => fetch(k).map(hash), get());
            if cachedValues.contains(value) {
                return value;
            }
            match cachedValues {
                List::Cons(cachedValue, _) => { return cachedValue; }
                List::Nil => {
                    let (newValue, deps) = track(task, fetch);
                    modify(
                        s => recordCT(key, newValue, deps.map((k, v) => (k, hash(v))), s)
                    );
                    return newValue;
                }
            }
        }
    };

// Fig 10. Code on 79:22 (contd.)
//
// -- Cloud Shake build system, implementation of ’suspending’ is given in Fig. 9
// cloudShake :: (Ord k, Hashable v) => Build Monad (CT k v) k v
// cloudShake = suspending ctRebuilder
//
// -- CloudBuild build system, implementation of ’topological’ is given in Fig. 7
// cloudBuild :: (Ord k, Hashable v) => Build Applicative (CT k v) k v
// cloudBuild = topological (adaptRebuilder ctRebuilder)
//
// -- Convert a monadic rebuilder to the corresponding applicative one
// adaptRebuilder :: Rebuilder Monad i k v -> Rebuilder Applicative i k v
// adaptRebuilder rebuilder key value task = rebuilder key value $ Task $ run task
//
// -- Buck build system, implementation of ’topological’ is given in Fig. 7
// buck :: (Ord k, Hashable v) => Build Applicative (DCT k v) k v
// buck = topological (adaptRebuilder dctRebuilder)
//
// -- Rebuilder based on deep constructive traces, analogous to ’ctRebuilder’
// dctRebuilder :: (Eq k, Hashable v) => Rebuilder Monad (DCT k v) k v
//
// -- Nix build system, implementation of ’suspending’ is given in Fig. 9
// nix :: (Ord k, Hashable v) => Build Monad (DCT k v) k v
// nix = suspending dctRebuilder

let cloudShake: Build<Monad, CT<K, V>, K, V>
    = suspending(ctRebuilder);

let cloudBuild: Build<Applicative, CT<K, V>, K, V>
    = topological(adaptRebuilder(ctRebuilder));

let adaptRebuilder: Fn(Rebuilder<Monad, I, K, V>) -> Rebuilder<Applicative, I, K, V> =
    rebuilder => {
        return (key, value, task) => rebuilder(key, value, Task { run: task.run });
    };

let buck: Build<Applicative, DCT<K, V>, K, V>
    = topological(adaptRebuilder(dctRebuilder));

let dctRebuilder: Rebuilder<Monad, DCT<K, V>, K: Eq, V: Hashable>;

let nix: Build<Monad, DCT<K, V>, K, V>
    = suspending(dctRebuilder);

// Code on 79:24
// sprsh3 :: Tasks MonadPlus String Integer
// sprsh3 "B1" = Just $ Task $ \fetch -> (+) <$> fetch "A1" <*> (pure 1 <|> pure 2)
// sprsh3 _ = Nothing
let spreadsheet3: Tasks<MonadPlus, String, Integer> =
    k => match k {
        // No easy translation here without going into details of Alternative/MonadPlus :(
        "B1" => Some(Task { run: fetch => fetch("A1") + eitherOr(1, 2) }),
        _    => None
    };

// Code on 79:25
// sprsh4 "B1" = Just $ Task $ \fetch -> do
//     formula <- fetch "B1-formula"
//     evalFormula fetch formula
let spreadsheet4: Tasks<Monad, String, Integer> =
    k => match k {
        "B1" => Some(Task {
            run: fetch => {
                let formula = fetch("B1-formula");
                return evalFormula(fetch, formula);
            }
        })
        ... // skipped
    };
