import { useEffect, useState } from "react";

type Severity = "ok" | "warn" | "critical";

type YoloEvent = {
  id: number;
  timestamp: string;
  severity: Severity;
  angle: number;
  message: string;
};

type WhisperEvent = {
  id: number;
  timestamp: string;
  text: string;
  kind: "info" | "pain" | "command";
};

function severityLabel(sev: Severity) {
  switch (sev) {
    case "ok":
      return "Ergonomia OK";
    case "warn":
      return "Pyöristyvä selkä";
              </div>
    case "critical":
      return "Väärä asento – keskeytä!";
  }
}

function severityColor(sev: Severity) {
  switch (sev) {
    case "ok":
      return "bg-emerald-500/20 text-emerald-300 border-emerald-500/60";
    case "warn":
      return "bg-amber-500/20 text-amber-300 border-amber-500/60";
    case "critical":
      return "bg-red-500/20 text-red-300 border-red-500/60";
  }
}

export default function App() {
  const [isRunning, setIsRunning] = useState(false);

  const [currentAngle, setCurrentAngle] = useState(10);
  const [currentSeverity, setCurrentSeverity] = useState<Severity>("ok");
  const [yoloEvents, setYoloEvents] = useState<YoloEvent[]>([]);

  const [lastUtterance, setLastUtterance] = useState<string>("(ei puhetta vielä)");
  const [whisperEvents, setWhisperEvents] = useState<WhisperEvent[]>([]);

  // MOCK: YOLO-data, pyörii vain kun isRunning = true
  useEffect(() => {
              </div>
    if (!isRunning) return;

    let id = 1;
    const interval = setInterval(() => {
      const angle = Math.round(10 + Math.random() * 50);
      let sev: Severity = "ok";
      let msg = "Asento hyvä";

      if (angle > 35 && angle <= 45) {
        sev = "warn";
        msg = "Selkä alkaa pyöristyä";
      } else if (angle > 45) {
        sev = "critical";
        msg = "Väärä nostoasento – riski hoitajalle";
      }

      const ts = new Date().toLocaleTimeString("fi-FI", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });

      setCurrentAngle(angle);
      setCurrentSeverity(sev);
      setYoloEvents((prev) => {
        const next: YoloEvent = { id: id++, timestamp: ts, severity: sev, angle, message: msg };
        const merged = [next, ...prev];
        return merged.slice(0, 15);
      });
    }, 3000);

    return () => clearInterval(interval);
  }, [isRunning]);

  // MOCK: Whisper-data, pyörii vain kun isRunning = true
  useEffect(() => {
    if (!isRunning) return;

              </div>
    let id = 1;
    const phrases: WhisperEvent["text"][] = [
      "Ready",
      "Auts, sattuu",
      "Nosto ok",
      "Voitko laskea minut hitaammin",
      "Oho, ote lipsahti",
      "Kaikki ok",
    ];

    const interval = setInterval(() => {
      const ts = new Date().toLocaleTimeString("fi-FI", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });

      const text = phrases[Math.floor(Math.random() * phrases.length)];
      let kind: WhisperEvent["kind"] = "info";
      if (text.toLowerCase().includes("sattuu") || text.toLowerCase().includes("auts")) {
        kind = "pain";
      } else if (text.toLowerCase().includes("ready")) {
        kind = "command";
      }

      setLastUtterance(text);
      setWhisperEvents((prev) => {
        const next: WhisperEvent = { id: id++, timestamp: ts, text, kind };
        const merged = [next, ...prev];
              </div>
        // Täysi transkriptio – max 100 viimeistä repliikkiä
        return merged.slice(0, 100);
      });
    }, 5000);

    return () => clearInterval(interval);
  }, [isRunning]);

  const handleToggle = () => {
    setIsRunning((prev) => !prev);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Yläpalkki */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur">
        <div className="mx-auto max-w-6xl px-4 py-3 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-xl bg-emerald-500 flex items-center justify-center text-slate-950 font-black">
              AI
            </div>
            <div>
              <h1 className="text-lg font-semibold">Potilassiirto – YOLO + Whisper</h1>
              <p className="text-xs text-slate-400">
                Reaaliaikainen ergonomia ja puhehavainnointi (demo)
              </p>
            </div>
          </div>

          {/* Yksi nappi käynnistämään molemmat */}
          <button
            onClick={handleToggle}
            className={`px-5 py-2 rounded-full text-sm font-semibold shadow-md border transition ${
              isRunning
                ? "bg-red-500 hover:bg-red-600 border-red-400"
                : "bg-emerald-500 hover:bg-emerald-600 border-emerald-400"
            }`}
          >
            {isRunning ? "Pysäytä YOLO + Whisper" : "Käynnistä YOLO + Whisper"}
          </button>
        </div>
      </header>

      {/* Pääosa: vasen = YOLO+video, oikea = Whisper */}
      <main className="flex-1 mx-auto max-w-6xl w-full px-4 py-6 grid gap-6 md:grid-cols-2">
        {/* Vasen: YOLO + videopaneeli */}
        <section className="flex flex-col gap-4">
          {/* YOLO-tila-kortti */}
          <div
            className={`rounded-2xl border px-6 py-4 shadow-lg flex items-center justify-between ${severityColor(
              currentSeverity
            )}`}
          >
            <div className="flex flex-col gap-1">
              <span className="text-xs uppercase tracking-wide opacity-80">
                YOLO – hoitajan asento
              </span>
              <span className="text-2xl font-bold">
                {severityLabel(currentSeverity)}
              </span>
              <span className="text-sm opacity-80">
                Viimeisin kulma:{" "}
                <span className="font-mono text-base">{currentAngle}°</span>
              </span>
            </div>
            <div className="flex flex-col items-end gap-1">
              <span className="text-xs opacity-70">Selkäkulma</span>
              <div className="text-4xl font-black font-mono">
                {currentAngle}°
              </div>
              <span className="text-[10px] uppercase tracking-wide opacity-60">
                {isRunning ? "Seuranta päällä" : "Seuranta pois päältä"}
              </span>
            </div>
          </div>

          {/* Video-placeholder */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/70 shadow-inner flex-1 flex flex-col">
            <div className="flex items-center justify-between px-4 py-2 border-b border-slate-800 text-xs text-slate-400">
              <span>Videovirta (YOLO pose / overlay)</span>
              <span className="font-mono">camera0 • 1280×720</span>
            </div>
            <div className="flex-1 flex items-center justify-center p-4">
              <div className="aspect-video w-full max-w-3xl border border-dashed border-slate-700 rounded-xl flex items-center justify-center text-slate-500 text-sm">
                {/* myöhemmin tähän oikea video-canvas / <img /> */}
                Videokuva tähän (WebSocket / stream)
              </div>
            </div>
          </div>
        </section>

        {/* Oikea: Whisper-näkymä */}
        <section className="flex flex-col gap-4">
          {/* Viimeisin puhe */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/80 px-6 py-4 shadow-lg flex flex-col gap-2">
            <span className="text-xs uppercase tracking-wide text-slate-400">
              Whisper – viimeisin puhe
            </span>
            <div className="text-lg font-mono">
              “{lastUtterance}”
            </div>
            <span className="text-[11px] text-slate-500">
              Tämä tulee myöhemmin oikeasta Whisper-mallista. Kaikki puhe näkyy alla olevassa täydessä
              transkriptiossa.
            </span>
          </div>

          {/* Whisper – täysi transkriptio */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4 flex-1 flex flex-col">
            <div className="flex items-center justify-between mb-3">
              <div>
                <h2 className="text-sm font-semibold">Whisper – täysi transkriptio</h2>
                <p className="text-xs text-slate-400">
                  Kaikki viimeisimmät repliikit • uusin ensin (max 100)
                </p>
              </div>
            </div>

            <div className="flex-1 overflow-auto pr-1 space-y-2 text-sm">
              {whisperEvents.length === 0 && (
                <div className="text-xs text-slate-500">
                  Ei puhehavainnointeja vielä…
                </div>
              )}

              {whisperEvents.map((ev) => (
                <div
                  key={ev.id}
                  className="rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-2 flex items-center justify-between gap-3"
                >
                  <div className="flex flex-col">
                    <span className="text-xs text-slate-400 font-mono">
                      {ev.timestamp}
                    </span>
                    <span className="text-sm">
                      “{ev.text}”
                    </span>
                  </div>
                  <div className="flex flex-col items-end gap-1">
                    <span
                      className={`text-[10px] uppercase tracking-wide px-2 py-0.5 rounded-full ${
                        ev.kind === "pain"
                          ? "bg-red-500/20 text-red-300"
                          : ev.kind === "command"
                          ? "bg-emerald-500/20 text-emerald-300"
                          : "bg-slate-700/60 text-slate-200"
                      }`}
                    >
                      {ev.kind === "pain"
                        ? "KIPU"
                        : ev.kind === "command"
                        ? "KOMENTO"
                        : "INFO"}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
