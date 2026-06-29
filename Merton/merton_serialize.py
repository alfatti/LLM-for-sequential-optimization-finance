"""
Serialization for Merton sequential-decision trajectories, in the schema style of
Paper 1 (Appendix D): per step  <S_t> price <W_t> wealth <A_t> action <R_t> return.

Rich state (price + wealth) per the user's choice; full trajectory is serialized so
the model conditions on the entire history. Action quantized to 2 decimals (Phase 0
showed quantization floor ~1e-3% — negligible). The hidden per-task scalar the model
must infer is c = action/wealth (CRRA: c=(mu-r)/(sigma^2 rho)).

The parser is the inverse map used at eval time to extract the model's action token.
"""
import re

A_DECIMALS = 2
SW_DECIMALS = 3
R_DECIMALS = 4

def fmt(v, d):
    return f"{v:.{d}f}"

def serialize_step(S, W, A, R):
    return (f"<S> {fmt(S, SW_DECIMALS)} <W> {fmt(W, SW_DECIMALS)} "
            f"<A> {fmt(A, A_DECIMALS)} <R> {fmt(R, R_DECIMALS)}")

def serialize_trajectory(prices, wealth, actions, returns):
    """prices,wealth length T+1 (states), actions,returns length T."""
    T = len(actions)
    parts = [serialize_step(prices[k], wealth[k], actions[k], returns[k]) for k in range(T)]
    # final realized wealth as a terminal marker (no action)
    parts.append(f"<S> {fmt(prices[T], SW_DECIMALS)} <W> {fmt(wealth[T], SW_DECIMALS)} <END>")
    return " ".join(parts)

# query: support context + partial query trajectory up to the current step, ending at "<A>"
def build_query_prefix(partial_prices, partial_wealth, partial_actions, partial_returns):
    """Steps already taken are fully serialized; the current step is opened up to '<A> '."""
    parts = []
    for k in range(len(partial_actions)):
        parts.append(serialize_step(partial_prices[k], partial_wealth[k],
                                     partial_actions[k], partial_returns[k]))
    cur = len(partial_actions)
    parts.append(f"<S> {fmt(partial_prices[cur], SW_DECIMALS)} "
                 f"<W> {fmt(partial_wealth[cur], SW_DECIMALS)} <A>")
    return " ".join(parts)

_action_re = re.compile(r"<A>\s*(-?\d+\.?\d*)")

def parse_first_action(text):
    """Extract the first numeric action following an <A> tag in model output."""
    m = _action_re.search(text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None

if __name__ == "__main__":
    s = serialize_trajectory([1.0,1.05,1.02],[1.0,1.08,1.13],[1.12,1.05],[0.08,0.046])
    print("TRAJECTORY:\n", s)
    q = build_query_prefix([1.0,1.05],[1.0,1.08],[1.12],[0.08])
    print("\nQUERY PREFIX (model completes after <A>):\n", q)
    print("\nparse test:", parse_first_action("<A> 1.27 <R> 0.03"),
          parse_first_action("the action is 0.84 maybe"))
