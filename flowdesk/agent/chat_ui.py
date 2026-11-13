"""Claude-Code-style chat widget for FlowDesk's Agent Chat dock.

Pure Qt (the module objects are passed in, so it works with PySide6 and PyQt alike).  The host supplies ``start_run(backend, prompt, callback)`` (see agentcli.AgentRun); events reach the
widget through a queued Qt signal so they may come from any thread.

Look & feel mirrors the Claude Code terminal UI: dark monospace transcript, ``>`` user prompts, ``⏺`` lines for
assistant text and tool calls, code windows for the commands and scripts the agent runs, ``⎿`` result previews,
an animated status line while the agent works, a bordered ``>`` input box with a slash-command popup, a model
switcher (provider, model, effort — like Claude Code's /model) and a shortcuts footer.
"""
from __future__ import annotations

import html
import json
import os
import re
import time

ORANGE = "#D97757"      # Claude's accent
BG = "#1F1F1F"
BG_INPUT = "#262626"
FG = "#E6E6E6"
DIM = "#8B8B8B"
GREEN = "#5FB369"
RED = "#E5484D"
BLUE = "#7AA2F7"
BORDER = "#3A3A3A"
CODE_BG = "#161616"
SPINNER = "·✢✳✶✻✽✻✶✳✢"
VERBS = ["Thinking", "Pondering", "Gating", "Inspecting", "Counting", "Reasoning", "Composing"]
SLASH = [
    ("/model", "choose provider, model and effort"),
    ("/clear", "new conversation"),
    ("/claude", "switch to Claude Code"),
    ("/codex", "switch to Codex"),
    ("/skills", "ask the agent to list its skills"),
    ("/stop", "interrupt the current run"),
    ("/help", "shortcuts and commands"),
]
HELP = ("/model   provider · model · effort (or the button bottom right)\n"
        "/claude  /codex   switch provider        /clear   new conversation\n"
        "/skills  list the agent's skill files     /stop    interrupt\n"
        "Enter sends · Shift+Enter newline · Esc interrupts · ↑/↓ history · Tab accepts a / suggestion")


def _enum(ns, group, name):
    """PyQt5 exposes Qt.Key_Return; PyQt6 exposes Qt.Key.Key_Return."""
    v = getattr(ns, name, None)
    if v is None:
        v = getattr(getattr(ns, group), name)
    return v


def _esc(s):
    return html.escape(str(s), quote=False)


def _md_inline(s):
    s = _esc(s)
    s = re.sub(r"`([^`\n]+)`", r'<code style="background:%s;padding:0 3px;border-radius:3px">\1</code>' % CODE_BG, s)
    s = re.sub(r"\*\*([^*\n]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<i>\1</i>", s)
    return s


def _code_window(code, lang="", title=""):
    """A code block like Claude Code shows for commands/scripts: dark box, line numbers, optional title."""
    lines = str(code).rstrip("\n").split("\n")
    rows = []
    for i, line in enumerate(lines, 1):
        rows.append('<tr><td style="color:%s;padding:0 10px 0 6px;text-align:right">%d</td>'
                    '<td style="white-space:pre;padding:0 8px 0 0">%s</td></tr>' % (DIM, i, _esc(line)))
    head = ('<div style="color:%s;margin:0 0 2px 14px">%s%s</div>' % (DIM, _esc(title), (" · " + lang) if lang else "")) if title else ""
    return (head + '<table cellpadding="1" cellspacing="0" style="background:%s;border:1px solid %s;margin:2px 0 4px 14px;'
            'font-family:inherit">%s</table>' % (CODE_BG, BORDER, "".join(rows)))


def _md_block(text):
    out = []
    parts = re.split(r"```([a-zA-Z0-9_+-]*)\n?", text)
    # re.split with a group yields [text, lang, code, text, lang, code, ...]
    i = 0
    while i < len(parts):
        chunk = parts[i]
        if i % 3 == 0:
            lines = [_md_inline(l) for l in chunk.strip("\n").split("\n")]
            if lines and any(lines):
                out.append("<br/>".join(lines))
            i += 1
        else:
            lang, code = parts[i], parts[i + 1] if i + 1 < len(parts) else ""
            out.append(_code_window(code, lang))
            i += 2
    return "".join(out)


def _preview(text, max_lines=2, width=110):
    text = str(text)
    stripped = text.strip()
    if stripped[:1] in "{[":
        try:
            obj = json.loads(stripped)
            compact = json.dumps(obj, ensure_ascii=False)
            total = len(text.splitlines())
            return ([compact if len(compact) <= width else compact[:width - 1] + "…"], max(0, total - 1))
        except Exception:
            pass
    lines = [l for l in text.splitlines() if l.strip()]
    shown = []
    for l in lines[:max_lines]:
        shown.append(l if len(l) <= width else l[:width - 1] + "…")
    return shown, len(lines) - len(shown)


def _tool_call_html(text):
    """'<server>/<tool> <json args>' -> headline + code window for commands / scripts, compact args otherwise."""
    name, _, args = text.partition(" ")
    args = args.strip()
    try:
        obj = json.loads(args) if args else {}
    except Exception:
        obj = None
    head = '<div><span style="color:%s">⏺</span> <b>%s</b>' % (GREEN, _esc(name))
    tool = name.rsplit("/", 1)[-1]
    if isinstance(obj, dict) and tool == "python" and obj.get("code") is not None:
        return head + "</div>" + _code_window(obj["code"], "python", "script")
    if isinstance(obj, dict) and tool == "save_skill" and obj.get("content") is not None:
        return head + '<span style="color:%s">(%s)</span></div>' % (DIM, _esc(obj.get("name", ""))) + _code_window(obj["content"], "markdown", "skill")
    return head + '<span style="color:%s">(%s)</span></div>' % (DIM, _esc(args[:400]))


def make_chat_widget(QtWidgets, QtCore, QtGui, *, app, pid, start_run, sessions, cwd=None,
                     find_backend=None, title="Agent Chat", agentcli=None):
    """Return a QWidget with .submit(text), .on_event(kind, payload), .set_backend(key), .run, .relay."""
    Signal = getattr(QtCore, "Signal", None) or getattr(QtCore, "pyqtSignal")
    Qt = QtCore.Qt
    K_RETURN, K_ENTER, K_TAB = _enum(Qt, "Key", "Key_Return"), _enum(Qt, "Key", "Key_Enter"), _enum(Qt, "Key", "Key_Tab")
    K_ESC, K_UP, K_DOWN = _enum(Qt, "Key", "Key_Escape"), _enum(Qt, "Key", "Key_Up"), _enum(Qt, "Key", "Key_Down")
    SHIFT = _enum(Qt, "KeyboardModifier", "ShiftModifier")
    mono = "Menlo, 'SF Mono', Consolas, 'DejaVu Sans Mono', monospace"
    labels = getattr(agentcli, "BACKEND_LABELS", {"claude": "Claude Code", "codex": "Codex"})
    model_choices = getattr(agentcli, "MODEL_CHOICES", {"claude": [("", "default")], "codex": [("", "default")]})
    effort_choices = getattr(agentcli, "EFFORT_CHOICES", {"claude": [""], "codex": [""]})

    class Relay(QtCore.QObject):
        event = Signal(str, str)

    class PromptEdit(QtWidgets.QPlainTextEdit):
        submitted = Signal()
        interrupted = Signal()
        history_prev = Signal()
        history_next = Signal()
        accept_suggestion = Signal()

        def keyPressEvent(self, ev):
            k = ev.key()
            if k in (K_RETURN, K_ENTER) and not (ev.modifiers() & SHIFT):
                self.submitted.emit()
                return
            if k == K_TAB:
                self.accept_suggestion.emit()
                return
            if k == K_ESC:
                self.interrupted.emit()
                return
            if k == K_UP and self.textCursor().blockNumber() == 0:
                self.history_prev.emit()
                return
            if k == K_DOWN and self.textCursor().blockNumber() == self.document().blockCount() - 1:
                self.history_next.emit()
                return
            super().keyPressEvent(ev)

    class ChatWidget(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.app, self.pid = app, pid
            self.available = (agentcli.available_backends() if agentcli else [b for b in ("claude", "codex") if (find_backend and find_backend(b))]) or []
            self.backend = (agentcli.default_backend() if agentcli else (self.available[0] if self.available else "claude"))
            if self.available and self.backend not in self.available:
                self.backend = self.available[0]
            self.blocks = []
            self.draft = None
            self.draft_text = ""
            self.run = None
            self.t0 = None
            self.history, self.hist_pos = [], 0
            self.relay = Relay()
            self.relay.event.connect(self.on_event)
            self.spin = 0
            self.verb = VERBS[0]
            self._last_stats = None
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self._tick)
            self.render_timer = QtCore.QTimer(self)
            self.render_timer.setSingleShot(True)
            self.render_timer.timeout.connect(self._render)
            self._build()
            self._welcome()

        # ---- settings helpers --------------------------------------------------
        def _settings(self):
            return agentcli.backend_settings(self.backend) if agentcli else {"model": "", "effort": ""}

        def _model_label(self):
            st = self._settings()
            m = st.get("model") or "default"
            e = st.get("effort")
            return "%s · %s%s" % (labels.get(self.backend, self.backend).lower(), m, (" · " + e) if e else "")

        # ---- layout ------------------------------------------------------------
        def _build(self):
            self.setMinimumSize(500, 440)
            self.setStyleSheet(
                "QWidget{background:%s;color:%s;font-family:%s;font-size:12.5px}"
                "QTextBrowser{background:%s;border:none;padding:6px 10px}"
                "QPlainTextEdit{background:%s;border:1px solid %s;border-radius:4px;padding:4px 6px;color:%s;selection-background-color:%s}"
                "QLabel{color:%s}"
                "QToolButton{background:%s;border:1px solid %s;border-radius:4px;padding:2px 8px;color:%s}"
                "QToolButton:hover{border-color:%s;color:%s}"
                "QToolButton::menu-indicator{image:none}"
                "QMenu{background:%s;color:%s;border:1px solid %s}"
                "QMenu::item:selected{background:%s;color:%s}"
                "QListWidget{background:%s;border:1px solid %s;color:%s}"
                "QListWidget::item:selected{background:%s;color:%s}"
                % (BG, FG, mono, BG, BG_INPUT, BORDER, FG, ORANGE, DIM, BG_INPUT, BORDER, FG, ORANGE, FG,
                   BG_INPUT, FG, BORDER, ORANGE, BG, BG_INPUT, BORDER, FG, ORANGE, BG))
            lay = QtWidgets.QVBoxLayout(self)
            lay.setContentsMargins(6, 6, 6, 6)
            lay.setSpacing(4)
            self.view = QtWidgets.QTextBrowser()
            self.view.setOpenExternalLinks(False)
            self.view.setReadOnly(True)
            lay.addWidget(self.view, 1)
            self.status = QtWidgets.QLabel("")
            self.status.setStyleSheet("color:%s;padding-left:4px" % ORANGE)
            self.status.hide()
            lay.addWidget(self.status)
            # slash-command suggestions (shown while the prompt starts with "/")
            self.suggest = QtWidgets.QListWidget()
            self.suggest.setMaximumHeight(120)
            self.suggest.hide()
            self.suggest.itemClicked.connect(lambda it: self._use_suggestion(it.data(32)))
            lay.addWidget(self.suggest)
            row = QtWidgets.QHBoxLayout()
            row.setSpacing(6)
            caret = QtWidgets.QLabel(">")
            caret.setStyleSheet("color:%s;font-weight:bold;padding-top:4px" % FG)
            caret.setAlignment(_enum(Qt, "AlignmentFlag", "AlignTop"))
            row.addWidget(caret)
            self.prompt = PromptEdit()
            self.prompt.setPlaceholderText('Try "gate the lymphocytes and report CD4/CD8 per donor" or "compensate the samples"   (/ for commands)')
            self.prompt.setFixedHeight(54)
            self.prompt.submitted.connect(self._on_submit)
            self.prompt.interrupted.connect(self.stop)
            self.prompt.history_prev.connect(lambda: self._history(-1))
            self.prompt.history_next.connect(lambda: self._history(+1))
            self.prompt.accept_suggestion.connect(self._accept_suggestion)
            self.prompt.textChanged.connect(self._update_suggestions)
            row.addWidget(self.prompt, 1)
            lay.addLayout(row)
            foot = QtWidgets.QHBoxLayout()
            self.foot_left = QtWidgets.QLabel("/help · /model")
            self.model_btn = QtWidgets.QToolButton()
            self.model_btn.setPopupMode(_enum(QtWidgets.QToolButton, "ToolButtonPopupMode", "InstantPopup"))
            self.model_btn.setToolTip("Provider, model and effort (like /model in Claude Code)")
            self.model_btn.setMenu(self._build_model_menu())
            self.foot_right = QtWidgets.QLabel("")
            foot.addWidget(self.foot_left)
            foot.addStretch(1)
            foot.addWidget(self.foot_right)
            foot.addWidget(self.model_btn)
            lay.addLayout(foot)
            self._update_footer()

        def _build_model_menu(self):
            menu = QtWidgets.QMenu(self)
            self._rebuild_model_menu(menu)
            menu.aboutToShow.connect(lambda: self._rebuild_model_menu(menu))
            return menu

        def _rebuild_model_menu(self, menu):
            menu.clear()
            st = self._settings()
            for b in ("claude", "codex"):
                installed = b in self.available
                sub = menu.addMenu("%s%s" % (labels[b], "" if installed else "  (not installed)"))
                sub.setEnabled(installed)
                if not installed:
                    continue
                sel_model = st["model"] if b == self.backend else (agentcli.backend_settings(b)["model"] if agentcli else "")
                sel_effort = st["effort"] if b == self.backend else (agentcli.backend_settings(b)["effort"] if agentcli else "")
                m_menu = sub.addMenu("Model")
                for value, label in model_choices.get(b, []):
                    act = m_menu.addAction(("● " if value == sel_model else "   ") + label)
                    act.triggered.connect(lambda _=False, b=b, v=value: self._choose(b, model=v))
                act = m_menu.addAction("   other… (type /model <name>)")
                act.setEnabled(False)
                e_menu = sub.addMenu("Effort")
                for value in effort_choices.get(b, [""]):
                    act = e_menu.addAction(("● " if value == sel_effort else "   ") + (value or "default"))
                    act.triggered.connect(lambda _=False, b=b, v=value: self._choose(b, effort=v))
                if b != self.backend:
                    act = sub.addAction("Use %s" % labels[b])
                    act.triggered.connect(lambda _=False, b=b: self._choose(b))
            if not self.available:
                act = menu.addAction("No Claude Code or Codex CLI found — see docs/mcp.md")
                act.setEnabled(False)

        def _choose(self, backend, model=None, effort=None):
            if agentcli:
                if model is not None or effort is not None:
                    agentcli.set_backend_settings(backend, model=model, effort=effort)
                agentcli.set_default_backend(backend)
            self.set_backend(backend)
            self.add_note("agent: %s" % self._model_label())

        def _update_footer(self):
            self.model_btn.setText(self._model_label() + " ▾")
            self.foot_right.setText('<span style="color:%s">%s pid %d</span>' % (DIM, "FlowDesk", self.pid))

        # ---- slash suggestions -------------------------------------------------
        def _update_suggestions(self):
            text = self.prompt.toPlainText()
            if text.startswith("/") and "\n" not in text and " " not in text:
                self.suggest.clear()
                for cmd, desc in SLASH:
                    if cmd.startswith(text):
                        it = QtWidgets.QListWidgetItem("%-10s %s" % (cmd, desc))
                        it.setData(32, cmd)
                        self.suggest.addItem(it)
                if self.suggest.count():
                    self.suggest.setCurrentRow(0)
                    self.suggest.show()
                    return
            self.suggest.hide()

        def _accept_suggestion(self):
            if self.suggest.isVisible() and self.suggest.currentItem():
                self._use_suggestion(self.suggest.currentItem().data(32))

        def _use_suggestion(self, cmd):
            self.prompt.setPlainText(cmd + " ")
            self.prompt.moveCursor(_enum(QtGui.QTextCursor, "MoveOperation", "End"))
            self.suggest.hide()
            self.prompt.setFocus()

        # ---- transcript blocks ----------------------------------------------
        def _welcome(self):
            where = cwd or os.getcwd()
            who = labels.get(self.backend, self.backend) if self.available else "No agent CLI found (install Claude Code or Codex)"
            box = ('<table width="100%%" cellpadding="7" cellspacing="0" style="border:1px solid %s;margin:2px 0 8px 0">'
                   '<tr><td style="border:1px solid %s">'
                   '<span style="color:%s">✻</span> <b>Welcome to %s</b><br/><br/>'
                   '<span style="color:%s">%s · connected to this FlowDesk window (pid %d)<br/>'
                   'cwd: %s<br/>Type a request below · /model to switch provider or model · /help for shortcuts</span></td></tr></table>'
                   % (BORDER, BORDER, ORANGE, _esc(title), DIM, _esc(who), self.pid, _esc(where)))
            self._add(box)

        def _add(self, html_block):
            self.blocks.append(html_block)
            self._schedule_render()
            return len(self.blocks) - 1

        def _schedule_render(self):
            if not self.render_timer.isActive():
                self.render_timer.start(30)

        def _render(self):
            doc = '<div style="font-family:%s;font-size:12.5px;color:%s;line-height:1.35">%s</div>' % (
                mono, FG, '<div style="margin:0 0 6px 0"></div>'.join(self.blocks))
            self.view.setHtml(doc)
            self._scroll_bottom()
            QtCore.QTimer.singleShot(0, self._scroll_bottom)

        def _scroll_bottom(self):
            try:
                cur = self.view.textCursor()
                cur.movePosition(_enum(QtGui.QTextCursor, "MoveOperation", "End"))
                self.view.setTextCursor(cur)
                self.view.ensureCursorVisible()
            except Exception:
                pass
            sb = self.view.verticalScrollBar()
            sb.setValue(sb.maximum())

        def add_user(self, text):
            self._add('<div style="color:%s;margin:6px 0 4px 0"><span style="color:%s">&gt;</span> %s</div>'
                      % (DIM, FG, _md_inline(text).replace("\n", "<br/>")))

        def add_assistant(self, text):
            self.draft = None
            self.draft_text = ""
            self._add('<div><span style="color:%s">⏺</span> %s</div>' % (ORANGE, _md_block(text)))

        def add_delta(self, delta):
            if self.draft is None:
                self.draft_text = ""
                self.draft = self._add("")
            self.draft_text += delta
            self.blocks[self.draft] = '<div><span style="color:%s">⏺</span> %s<span style="color:%s">▌</span></div>' % (
                ORANGE, _md_block(self.draft_text), DIM)
            self._schedule_render()

        def finalize_draft(self, text):
            if self.draft is not None:
                self.blocks[self.draft] = '<div><span style="color:%s">⏺</span> %s</div>' % (ORANGE, _md_block(text))
                self.draft = None
                self.draft_text = ""
                self._schedule_render()
            else:
                self.add_assistant(text)

        def add_tool(self, text):
            self.draft = None
            self._add(_tool_call_html(text))

        def add_tool_result(self, text):
            shown, more = _preview(text)
            tail = ('<br/>&nbsp;&nbsp;&nbsp;… +%d lines' % more) if more > 0 else ""
            self._add('<div style="color:%s;margin-left:14px">⎿&nbsp; %s%s</div>'
                      % (DIM, "<br/>&nbsp;&nbsp;&nbsp;".join(_esc(l) for l in shown) or "(no output)", tail))

        def add_note(self, text, color=DIM):
            self._add('<div style="color:%s;white-space:pre-wrap">%s</div>' % (color, _esc(text)))

        # ---- running -----------------------------------------------------------
        def set_backend(self, key):
            if key in labels:
                self.backend = key
                self._update_footer()

        def submit(self, text):
            text = (text or "").strip()
            if not text:
                return
            if text.startswith("/"):
                return self._slash(text)
            if self.run is not None:
                self.add_note("(still working — Esc to interrupt)")
                return
            self.history.append(text)
            self.hist_pos = len(self.history)
            self.add_user(text)
            try:
                self.run = start_run(self.backend, text, lambda k, p: self.relay.event.emit(k, p if isinstance(p, str) else json.dumps(p)))
            except Exception as e:
                self.add_note("! %s" % e, RED)
                self.run = None
                return
            self.t0 = time.time()
            self.verb = VERBS[len(self.history) % len(VERBS)]
            self.status.show()
            self.timer.start(120)
            self.prompt.setFocus()

        def _slash(self, text):
            cmd, _, rest = text.partition(" ")
            cmd = cmd.lower()
            rest = rest.strip()
            if cmd == "/clear":
                sessions[self.backend] = None
                self.blocks = []
                self._welcome()
                self.add_note("conversation cleared")
            elif cmd in ("/claude", "/codex"):
                self._choose(cmd[1:])
                if rest:
                    self.submit(rest)
            elif cmd == "/model":
                if rest:
                    parts = rest.split()
                    backend = self.backend
                    model = effort = None
                    for ptok in parts:
                        if ptok in ("claude", "codex"):
                            backend = ptok
                        elif ptok in effort_choices.get(backend, []) and ptok:
                            effort = ptok
                        elif ptok in ("default", "none"):
                            model = ""
                        else:
                            model = ptok
                    self._choose(backend, model=model, effort=effort)
                else:
                    self.add_note("current: %s\nusage: /model [claude|codex] [<model>|default] [low|medium|high|xhigh]\n"
                                  "or use the %s ▾ button" % (self._model_label(), self._model_label()))
            elif cmd == "/skills":
                self.submit("List your available skills (call the skills tool) and summarise them in one line each.")
            elif cmd == "/stop":
                self.stop()
            elif cmd in ("/help", "/?"):
                self.add_note(HELP)
            else:
                self.add_note("unknown command %s — /help" % cmd, RED)

        def _on_submit(self):
            if self.suggest.isVisible() and self.suggest.currentItem() and self.prompt.toPlainText().strip() != self.suggest.currentItem().data(32):
                self._accept_suggestion()
                return
            text = self.prompt.toPlainText()
            self.prompt.clear()
            self.suggest.hide()
            self.submit(text)

        def _history(self, step):
            if not self.history:
                return
            self.hist_pos = max(0, min(len(self.history), self.hist_pos + step))
            self.prompt.setPlainText(self.history[self.hist_pos] if self.hist_pos < len(self.history) else "")
            self.prompt.moveCursor(_enum(QtGui.QTextCursor, "MoveOperation", "End"))

        def stop(self):
            if self.run is not None:
                self.run.cancel()
                self.add_note("interrupted", RED)

        def _tick(self):
            self.spin = (self.spin + 1) % len(SPINNER)
            secs = int(time.time() - self.t0) if self.t0 else 0
            self.status.setText('<span style="color:%s">%s %s…</span> <span style="color:%s">(esc to interrupt · %ds)</span>'
                                % (ORANGE, SPINNER[self.spin], self.verb, DIM, secs))

        def _finish(self, stats=None):
            self.timer.stop()
            self.status.hide()
            secs = time.time() - self.t0 if self.t0 else 0
            extra = ""
            if stats:
                cost = stats.get("total_cost_usd")
                usage = stats.get("usage") or {}
                toks = usage.get("output_tokens") or stats.get("output_tokens")
                if cost is not None:
                    extra += " · $%.2f" % cost
                if toks:
                    extra += " · %s output tokens" % toks
            self.add_note("✓ %s · %.1fs%s" % (self._model_label(), secs, extra))
            self.run = None
            self.t0 = None
            self.prompt.setFocus()

        # ---- events from the worker thread (via queued signal) ---------------
        def on_event(self, kind, payload):
            if kind == "delta":
                self.add_delta(payload)
            elif kind == "text":
                self.finalize_draft(payload)
            elif kind == "tool":
                self.add_tool(payload)
            elif kind == "tool_result":
                self.add_tool_result(payload)
            elif kind == "error":
                self.add_note("! " + payload, RED)
            elif kind == "stats":
                try:
                    self._last_stats = json.loads(payload)
                except Exception:
                    self._last_stats = None
            elif kind == "done":
                if self.draft is not None and payload:
                    self.finalize_draft(payload)
                self._finish(self._last_stats)
                self._last_stats = None

    return ChatWidget()
