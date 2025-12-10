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

  const [lastUtterance, setLastUtterance] = useState<string>(
    "(ei puhetta vielä)"
  );
  const [whisperEvents, setWhisperEvents] = useState<WhisperEvent[]>([]);

  // YOLO-kuva backendistä
  const [yoloImage, setYoloImage] = useState<string | null>(null);

  // Paneelien leveys: YOLO-paneelin prosenttiosuus (Whisper = 100 - value)
  const [layoutRatio, setLayoutRatio] = useState(55); // 55% YOLO / 45% Whisper

  // 🔹 YOLO: vastaanota dataa WebSocket-palvelimelta
  useEffect(() => {
    if (!isRunning) return;

    let idCounter = 1;
    const ws = new WebSocket("ws://localhost:8765");

    ws.onopen = () => {
      console.log("YOLO WebSocket connected");
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        // data: { source, timestamp, frame_idx, angle, severity, incorrect, frame }
        const angle = Number(data.angle ?? 0);
        const severity = (data.severity as Severity) ?? "ok";
        const timestamp = (data.timestamp as string) ?? "";

        const message =
          severity === "ok"
            ? "Asento hyvä"
            : severity === "warn"
            ? "Selkä alkaa pyöristyä"
            : "Väärä nostoasento – riski hoitajalle";

        setCurrentAngle(Math.round(angle));
        setCurrentSeverity(severity);

        setYoloEvents((prev) => {
          const next: YoloEvent = {
            id: idCounter++,
            timestamp,
            severity,
            angle: Math.round(angle),
            message,
          };
          const merged = [next, ...prev];
          return merged.slice(0, 15);
        });

        if (data.frame) {
          setYoloImage("data:image/png;base64," + data.frame);
        }
      } catch (err) {
        console.error("Invalid YOLO message", err);
      }
    };

    ws.onerror = (err) => {
      console.error("YOLO WebSocket error", err);
    };

    ws.onclose = () => {
      console.log("YOLO WebSocket closed");
    };

    return () => {
      ws.close();
    };
  }, [isRunning]);

  // 🔹 Whisper-mock: pyörii vain kun isRunning = true
  useEffect(() => {
    if (!isRunning) return;

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
      if (
        text.toLowerCase().includes("sattuu") ||
        text.toLowerCase().includes("auts")
      ) {
        kind = "pain";
      } else if (text.toLowerCase().includes("ready")) {
        kind = "command";
      }

      setLastUtterance(text);
      setWhisperEvents((prev) => {
        const next: WhisperEvent = { id: id++, timestamp: ts, text, kind };
        const merged = [next, ...prev];
        return merged.slice(0, 100);
      });
    }, 5000);

    return () => clearInterval(interval);
  }, [isRunning]);

  const handleToggle = () => {
    setIsRunning((prev) => !prev);
  };

  const handleLayoutChange = (value: number) => {
    // rajoitetaan väliin 35–65 %, ettei toinen paneeli katoa kokonaan
    const clamped = Math.min(65, Math.max(35, value));
    setLayoutRatio(clamped);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Yläpalkki */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur">
        <div className="w-full px-12 py-3 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-xl bg-emerald-500 flex items-center justify-center text-slate-950 font-black">
              AI
            </div>
            <div>
              <h1 className="text-lg font-semibold">
                Potilassiirto
              </h1>
              <p className="text-xs text-slate-400">
                Reaaliaikainen ergonomia ja puhehavainnointi (DEMO)
              </p>
            </div>
          </div>

          <button
            onClick={handleToggle}
            className={`px-5 py-2 rounded-full text-sm font-semibold shadow-md border transition ${
              isRunning
                ? "bg-red-500 hover:bg-red-600 border-red-400"
                : "bg-emerald-500 hover:bg-emerald-600 border-emerald-400"
            }`}
          >
            {isRunning ? "Pysäytä harjoitus" : "Käynnistä harjoitus"}
          </button>
        </div>
      </header>

      {/* Paneelien koon säätö */}
      <div className="mx-auto max-w-6xl px-4 pt-3">
        <label className="flex items-center gap-3 text-xs text-slate-400">
          Paneelien koko:
          <input
            type="range"
            min={35}
            max={65}
            value={layoutRatio}
            onChange={(e) => handleLayoutChange(Number(e.target.value))}
            className="w-40 accent-emerald-500"
          />
          <span className="tabular-nums text-slate-300">
            {layoutRatio}% YOLO / {100 - layoutRatio}% Whisper
          </span>
        </label>
      </div>

      {/* Pääosa: kaksi saraketta, leveys säädettävissä sliderilla */}
      <main className="flex-1 w-full px-4 py-4">
        <div
          className="w-full px-12 py-3"
          style={{
            display: "grid",
            gridTemplateColumns: `minmax(320px, ${layoutRatio}%) minmax(320px, ${
              100 - layoutRatio
            }%)`,
            columnGap: "1.5rem", // vastaa gap-6
          }}
        >
          {/* Vasen: YOLO */}
          <section className="h-full flex flex-col gap-4">
            {/* YOLO-tila-kortti */}
            <div
              className={`rounded-2xl border px-6 py-4 shadow-lg flex items-center justify-between ${severityColor(
                currentSeverity
              )}`}
            >
              <div className="flex flex-col gap-1">
                <span className="text-xs uppercase tracking-wide opacity-80">
                  Hoitajan asento
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

            {/* YOLO-kuva backendistä */}
            <div className="rounded-2xl border border-slate-800 bg-slate-900/70 shadow-inner flex-1 flex flex-col min-h-[260px]">
              <div className="flex items-center justify-between px-4 py-2 border-b border-slate-800 text-xs text-slate-400">
                <span>YOLO-videovirta (backend → WebSocket)</span>
                <span className="font-mono">annotated pose</span>
              </div>
              <div className="flex-1 flex flex-col items-center justify-center p-4 gap-2">
                <div className="aspect-video w-full border border-slate-700 rounded-xl overflow-hidden bg-black flex items-center justify-center">
                  {yoloImage ? (
                    <img
                      src={yoloImage}
                      className="w-full h-full object-cover"
                      alt="YOLO Stream"
                    />
                  ) : (
                    <span className="text-slate-500 text-sm">
                      YOLO-kuvaa ei vielä vastaanotettu…
                    </span>
                  )}
                </div>
                <div className="text-xs text-slate-500">
                  Kuva piirretään Pythonissa (YOLOv8 + pose) ja lähetetään
                  WebSocketin kautta.
                </div>
              </div>
            </div>
          </section>

          {/* Oikea: Whisper */}
          <section className="h-full flex flex-col gap-4">
            {/* Viimeisin puhe */}
            <div className="rounded-2xl border border-slate-800 bg-slate-900/80 px-6 py-4 shadow-lg flex flex-col gap-2">
              <span className="text-xs uppercase tracking-wide text-slate-400">
                Whisper – viimeisin puhe
              </span>
              <div className="text-lg font-mono">“{lastUtterance}”</div>
              <span className="text-[11px] text-slate-500">
                Tämä tulee myöhemmin oikeasta Whisper-mallista. Kaikki puhe
                näkyy alla olevassa täydessä transkriptiossa.
              </span>
            </div>

            {/* Whisper – täysi transkriptio */}
            <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4 flex-1 flex flex-col min-h-[260px]">
              <div className="flex items-center justify-between mb-3">
                <div>
                  <h2 className="text-sm font-semibold">
                    Whisper – täysi transkriptio
                  </h2>
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
                      <span className="text-sm">“{ev.text}”</span>
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
        </div>
      </main>
    </div>
  );
}
