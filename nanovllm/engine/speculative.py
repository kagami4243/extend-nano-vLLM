from dataclasses import dataclass


@dataclass(frozen=True)
class GreedyVerification:
    accepted_count: int
    accepted_tokens: list[int]
    replacement_token: int | None


def verify_greedy_proposals(
    proposal_tokens: list[int], target_tokens: list[int]
) -> GreedyVerification:
    if len(target_tokens) < len(proposal_tokens):
        raise ValueError("target verification must cover every proposal")
    accepted_count = 0
    for proposal, target in zip(proposal_tokens, target_tokens):
        if proposal != target:
            return GreedyVerification(
                accepted_count,
                proposal_tokens[:accepted_count],
                target,
            )
        accepted_count += 1
    replacement = (
        target_tokens[accepted_count]
        if len(target_tokens) > accepted_count
        else None
    )
    return GreedyVerification(accepted_count, proposal_tokens.copy(), replacement)
