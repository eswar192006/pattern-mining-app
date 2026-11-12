# Five-minute local demo

Run `./start_demo.sh` from the project root. It starts the API and Streamlit in **replay mode**. The six saved prompts below work without a live model or internet access; each response still runs the project's mining algorithm on the local dataset. Type each question into the chat. The rare question intentionally asks for clarification; choose the option to lower the minimum count to continue.

| Time | What to show | Question to type |
|---|---|---|
| 0:00–0:30 | Welcome screen; explain that Ollama normally translates language while the algorithms count locally. Replay mode substitutes saved interpretations. | — |
| 0:30–1:10 | Frequent pairs, overall. Point at the result table and exact request panel. | `Which products are usually bought together?` |
| 1:10–1:50 | A weekday filter. Point out that only Sunday orders are selected. | `Which products sell together on Sundays?` |
| 1:50–2:35 | Rare but strongly linked pairs, limited to Germany. The default rare threshold cannot return enough German baskets, so the app asks what to adjust. Choose `Lower the minimum count to 2 or less`. | `Find rare but closely linked pairs in Germany.` |
| 2:35–3:25 | Revenue based high-utility pairs. Explain that revenue is not profit; this question uses Java and SPMF. | `Top 5 pairs that earn over 20000.` |
| 3:25–4:05 | A seasonal filter across December and January. | `Find common bundles in December and January.` |
| 4:05–4:45 | Include a specific product in every result. | `What sells together with the Regency Cakestand 3 Tier?` |
| 4:45–5:00 | Recap: saved language responses keep the demo reliable; the tables and figures are freshly computed from the local data. | — |

If a mining query is slow, point to the visible progress state and use the remaining time to describe the request and filter details. For a live Ollama walkthrough, start Ollama with the configured model and use `./start_demo.sh --live`.

## Presentation assets

- Screenshots: to be added by the presenter.
- Backup video: to be recorded by the presenter.
