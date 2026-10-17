# Handoff template (docs/claude/STATUS.md)

Copy the skeleton, fill it, delete what does not apply. Target one page.

````markdown
# Status: <topic>

**Written:** <date, time zone> at commit <short hash> on branch <branch>
**One line:** <what is done, what is in flight>

## Start message for the next session
```
<output of handoff_start_message(...)>
```

## Objective
<The user's goal in their terms. 2-3 sentences.>

## State
- Done and verified: <claim> (evidence: <commit / file:line / NB-id>)
- Done but unverified: <claim>
- In progress: <what> (next action: <command>)
- Blocked: <what>, waiting on <whom>
- Live runs: | host | job id | what | read-only check command |

## Decisions
- <Chose X over Y because ...> (<SI-id or NB-id>)

## Dead ends
- <Tried X; failed because ...>. Do not retry unless <what would change>.

## Next steps
1. <Concrete action naming files, parameters, commands>

## Open questions for the user
- <Question that blocks a real choice>
````

Usable versus vague: "Continue improving the model" strands the reader. "Re-fit with `alpha=0.3` in `fit_model.py` (0.1 underfit, see Dead ends); compare against `baseline_fit.csv`" can be executed.
