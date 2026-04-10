"use client";

import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Activity, 
  Terminal as TerminalIcon, 
  Users, 
  AlertTriangle, 
  Play, 
  RotateCcw, 
  Clock, 
  Award,
  ChevronRight,
  Stethoscope
} from "lucide-react";
import { TriageState, Patient } from "../types";

// ─── Components ─────────────────────────────────────────────────────────

const Report = ({ state, onClose }: { state: TriageState, onClose: () => void }) => {
  const getGrade = (score: number) => {
    if (score >= 0.9) return { l: 'A', c: 'text-terminal-green', m: 'ELITE PERFORMANCE' };
    if (score >= 0.8) return { l: 'B', c: 'text-cyan', m: 'ADVANCED CLINICAL' };
    if (score >= 0.7) return { l: 'C', c: 'text-neon-orange', m: 'COMPETENT TRIAGE' };
    return { l: 'F', c: 'text-crimson', m: 'CRITICAL FAILURE' };
  };

  const grade = getGrade(state.reward);

  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
    >
      <div className="bg-background border-4 border-border p-8 max-w-2xl w-full relative overflow-hidden">
        {/* Background watermark */}
        <div className="absolute top-0 right-0 text-[120px] font-black opacity-5 -mr-8 -mt-8 rotate-12 select-none">
          {grade.l}
        </div>

        <div className="flex justify-between items-start mb-8">
          <div>
            <h2 className="text-3xl font-black text-white uppercase tracking-tighter">Clinical Audit Report</h2>
            <p className="text-xs opacity-50">STATION_ID: ALPHA-09 // EPISODE: {state.task_id.toUpperCase()}</p>
          </div>
          <div className="text-right">
            <span className={`text-6xl font-black ${grade.c}`}>{grade.l}</span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-8 mb-8">
          <div className="space-y-4">
            <div>
              <span className="text-[10px] uppercase opacity-50 block">Audit Score</span>
              <span className={`text-3xl font-bold ${grade.c}`}>{(state.reward * 100).toFixed(1)}%</span>
            </div>
            <div>
              <span className="text-[10px] uppercase opacity-50 block">Time to Disposition</span>
              <span className="text-xl font-bold text-white">{state.elapsed_minutes} Minutes</span>
            </div>
            <div>
              <span className="text-[10px] uppercase opacity-50 block">Clinical Assessment</span>
              <span className={`text-sm font-bold uppercase ${grade.c}`}>{grade.m}</span>
            </div>
          </div>

          <div className="border-l border-border pl-8 space-y-4">
            <div>
              <span className="text-[10px] uppercase opacity-50 block">Steps Executed</span>
              <span className="text-xl font-bold text-white">{state.step_number}</span>
            </div>
            <div>
              <span className="text-[10px] uppercase opacity-50 block">Resource Efficiency</span>
              <span className="text-xl font-bold text-white">OPTIMAL</span>
            </div>
            <div className="pt-4 border-t border-border/30">
              <p className="text-[10px] opacity-70 leading-relaxed italic text-terminal-green">
                "Subject demonstrated {grade.l === 'F' ? 'significant' : 'minimal'} deviation from established ESI guidelines. {grade.l === 'A' ? 'Exceptional foresight in critical window management.' : 'Review protocols for time-sensitive interventions.'}"
              </p>
            </div>
          </div>
        </div>

        <div className="flex flex-col gap-3">
          <button 
            onClick={onClose}
            className="w-full py-4 bg-terminal-green text-black font-black uppercase hover:bg-white transition-colors flex items-center justify-center gap-2"
          >
            <RotateCcw size={18} />
            Initialize New Shift
          </button>
          <div className="flex justify-center text-[8px] opacity-30 tracking-widest uppercase">
            Official Record // Confidential Medical Data // Department of AI Health
          </div>
        </div>
      </div>
    </motion.div>
  );
};

const Panel = ({ title, children, className = "", icon: Icon }: any) => (
  <div className={`flex flex-col border-2 border-border bg-glass overflow-hidden ${className}`}>
    <div className="bg-border px-3 py-1 flex items-center gap-2 text-xs font-bold uppercase text-white">
      {Icon && <Icon size={14} />}
      {title}
    </div>
    <div className="flex-1 overflow-auto relative">
      {children}
    </div>
  </div>
);

const Sparkline = ({ values, color }: { values: number[], color: string }) => {
  if (values.length < 2) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const points = values.map((v, i) => `${(i / (values.length - 1)) * 100},${100 - ((v - min) / range) * 100}`).join(' ');

  return (
    <div className="h-6 w-16 opacity-50">
      <svg viewBox="0 0 100 100" className="w-full h-full overflow-visible">
        <polyline
          fill="none"
          stroke={color}
          strokeWidth="4"
          points={points}
        />
      </svg>
    </div>
  );
};

const VitalLine = ({ label, value, unit, trend, isCritical, history }: any) => (
  <div className={`flex flex-col p-2 border border-border/50 ${isCritical ? 'bg-crimson/20 border-crimson animate-pulse' : ''}`}>
    <div className="flex justify-between items-start">
      <span className="text-[10px] uppercase opacity-70">{label}</span>
      {history && <Sparkline values={history} color={isCritical ? '#DC143C' : '#00FFFF'} />}
    </div>
    <div className="flex items-baseline gap-2">
      <span className={`text-2xl font-bold ${isCritical ? 'text-crimson' : ''}`}>{value}</span>
      <span className="text-xs opacity-50">{unit}</span>
      {trend && (
        <span className={`text-xs ${trend === '↑' ? 'text-crimson' : trend === '↓' ? 'text-cyan' : ''}`}>
          {trend}
        </span>
      )}
    </div>
  </div>
);

const MatrixRain = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789$+-*/=%\"'#&_(),.;:?!\\|{}<>[]^~";
    const fontSize = 14;
    const columns = canvas.width / fontSize;
    const drops: number[] = [];

    for (let i = 0; i < columns; i++) {
      drops[i] = 1;
    }

    const draw = () => {
      ctx.fillStyle = "rgba(5, 5, 5, 0.05)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.fillStyle = "#00FF41";
      ctx.font = `${fontSize}px monospace`;

      for (let i = 0; i < drops.length; i++) {
        const text = characters.charAt(Math.floor(Math.random() * characters.length));
        ctx.fillText(text, i * fontSize, drops[i] * fontSize);

        if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) {
          drops[i] = 0;
        }
        drops[i]++;
      }
    };

    const interval = setInterval(draw, 33);
    return () => clearInterval(interval);
  }, []);

  return <canvas ref={canvasRef} className="fixed inset-0 z-0 opacity-20 pointer-events-none" />;
};

// ─── Main Page ──────────────────────────────────────────────────────────

export default function Dashboard() {
  const [state, setState] = useState<TriageState | null>(null);
  const [showReport, setShowReport] = useState(false);
  const [thoughts, setThoughts] = useState<string[]>([]);
  const [selectedPatientId, setSelectedPatientId] = useState<string | null>(null);
  const [vitalsHistoryMap, setVitalsHistoryMap] = useState<Record<string, Record<string, number[]>>>({});
  const [logs, setLogs] = useState<{time: string, msg: string, type?: string}[]>([]);
  const [connected, setConnected] = useState(false);
  const ws = useRef<WebSocket | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    connectWs();
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      ws.current?.close();
    };
  }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [thoughts]);

  const connectWs = () => {
    // Don't stack connections
    if (ws.current && ws.current.readyState === WebSocket.CONNECTING) return;
    if (ws.current && ws.current.readyState === WebSocket.OPEN) return;

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    let host = window.location.host;
    // Dev mode: redirect port 3000 → backend port 7860
    if (host.includes("localhost:3000") || host.includes("127.0.0.1:3000")) {
        host = host.replace(":3000", ":7860");
    } else if (!host.includes(":") && window.location.port === "") {
        host = `${window.location.hostname}:7860`;
    }

    try {
      ws.current = new WebSocket(`${protocol}//${host}/ws`);
    } catch {
      scheduleReconnect();
      return;
    }

    ws.current.onopen = () => {
      setConnected(true);
      addLog("System", "Neural uplink established. Triage-OS v2.0 online.");
    };

    ws.current.onerror = () => {
      // onerror always fires before onclose; just log it
      console.warn("WebSocket error — backend may be offline");
    };

    ws.current.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === "observation") {
        setState(msg.data);
        if (msg.data.done) setShowReport(true);

        const incomingPatients = msg.data.patients as Patient[];
        setVitalsHistoryMap(prev => {
            const next = { ...prev };
            incomingPatients.forEach(p => {
                const id = p.patient_id;
                const h = next[id] || { HR: [], BP: [], SpO2: [], GCS: [] };
                next[id] = {
                    HR: [...h.HR, p.vitals.heart_rate].slice(-20),
                    BP: [...h.BP, p.vitals.systolic_bp].slice(-20),
                    SpO2: [...h.SpO2, p.vitals.spo2].slice(-20),
                    GCS: [...h.GCS, p.vitals.gcs].slice(-20),
                };
            });
            return next;
        });

        if (incomingPatients.length > 0) {
            setSelectedPatientId(prev => prev ?? incomingPatients[0].patient_id);
        }

        if (msg.data.last_action_result) addLog("Simulation", msg.data.last_action_result);
        if (msg.data.last_action_error) addLog("Error", msg.data.last_action_error, "error");
      } else if (msg.type === "token") {
        setThoughts(prev => [...prev, msg.content]);
      }
    };

    ws.current.onclose = () => {
      setConnected(false);
      addLog("System", "Neural link severed. Retrying in 5s...", "error");
      scheduleReconnect();
    };
  };

  const scheduleReconnect = () => {
    if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    reconnectTimer.current = setTimeout(connectWs, 5000);
  };

  const addLog = (source: string, msg: string, type?: string) => {
    const time = new Date().toLocaleTimeString('en-US', { hour12: false });
    setLogs(prev => [{ time, msg: `${source}: ${msg}`, type }, ...prev].slice(0, 50));
  };

  const sendCommand = (type: string, payload: any) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ type, ...payload }));
    }
  };

  const reset = (taskId: string) => {
    setThoughts([]);
    setShowReport(false);
    sendCommand("reset", { task_id: taskId });
    addLog("User", `Requesting re-initialization: ${taskId}`);
  };

  const activePatient = state?.patients.find(p => p.patient_id === selectedPatientId) || state?.patients[0];
  const activeHistory = selectedPatientId ? vitalsHistoryMap[selectedPatientId] : null;

  return (
    <main className="h-screen w-screen p-3 grid grid-cols-[1fr_450px_400px] grid-rows-[60px_1fr_150px] gap-3 bg-background relative">
      <MatrixRain />
      {state && showReport && <Report state={state} onClose={() => setShowReport(false)} />}
      {/* Header */}
      <header className="col-span-3 border-2 border-border bg-glass flex items-center justify-between px-6">
        <div className="flex items-center gap-4">
          <div className="w-3 h-3 rounded-full bg-terminal-green animate-pulse" />
          <h1 className="text-xl font-black tracking-widest text-white">
            AUTONOMOUS CLINICAL TRIAGE // V2.0 // DEEP_THOUGHT_REASONER
          </h1>
        </div>
        <div className="flex gap-8 text-sm">
          <div className="flex flex-col">
            <span className="text-[10px] opacity-50 uppercase">Session Status</span>
            <span className={connected ? "text-terminal-green" : "text-crimson"}>
              {connected ? "LINK_STABLE" : "LINK_OFFLINE"}
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-[10px] opacity-50 uppercase">Task Identity</span>
            <span className="text-cyan">{state?.task_id || "AWAITING_INIT"}</span>
          </div>
          <div className="flex flex-col">
            <span className="text-[10px] opacity-50 uppercase">Resource Load</span>
            <span className="text-neon-orange">78% CAPACITY</span>
          </div>
        </div>
      </header>

      {/* Left Panel: Waiting Room */}
      <Panel title="Waiting Room / Incoming Dispatch" icon={Users} className="row-span-1">
        <div className="p-2 flex flex-col gap-2">
          <AnimatePresence>
            {state?.patients.map((p) => (
              <motion.div
                key={p.patient_id}
                initial={{ x: -20, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                onClick={() => setSelectedPatientId(p.patient_id)}
                className={`p-3 border border-border bg-black/40 relative hover:border-cyan cursor-pointer group transition-all ${activePatient?.patient_id === p.patient_id ? 'border-cyan border-2 bg-cyan/5 shadow-[0_0_15px_rgba(0,255,255,0.1)]' : ''}`}
              >
                <div className="flex justify-between items-start mb-1">
                  <span className="text-xs font-bold text-cyan">{p.patient_id}</span>
                  <span className="text-[10px] opacity-50">{p.age}{p.sex}</span>
                </div>
                <div className="text-sm font-bold text-white group-hover:text-cyan transition-colors truncate">
                  {p.chief_complaint}
                </div>
                <div className="mt-2 flex gap-3 text-[10px] opacity-70">
                  <span>HR: {p.vitals.heart_rate}</span>
                  <span>BP: {p.vitals.systolic_bp}/{p.vitals.diastolic_bp}</span>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          {!state && (
            <div className="h-full flex items-center justify-center opacity-30 italic text-sm">
              Simulation not initialized.
            </div>
          )}
        </div>
      </Panel>

      {/* Center Panel: Live Vitals */}
      <Panel title="Active Patient Management / Vitals Stream" icon={Activity}>
        {activePatient ? (
          <div className="p-4 h-full flex flex-col">
            <div className="flex justify-between items-end mb-6">
              <div>
                <h2 className="text-2xl font-bold text-white border-b-2 border-cyan pb-1 inline-block">
                  {activePatient.chief_complaint}
                </h2>
                <p className="text-xs opacity-50 mt-1">ID: {activePatient.patient_id} // AGE: {activePatient.age} // SEX: {activePatient.sex}</p>
              </div>
              <div className="text-right">
                <span className="text-[10px] opacity-50 block uppercase">Reward Entropy</span>
                <span className="text-2xl font-bold text-terminal-green">{(state?.reward || 0).toFixed(3)}</span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <VitalLine 
                label="Heart Rate" 
                value={activePatient.vitals.heart_rate} 
                unit="BPM" 
                trend={activePatient.vitals_trend?.HR}
                history={activeHistory?.HR}
                isCritical={activePatient.vitals.heart_rate > 120 || activePatient.vitals.heart_rate < 50}
              />
              <VitalLine 
                label="Blood Pressure" 
                value={`${activePatient.vitals.systolic_bp}/${activePatient.vitals.diastolic_bp}`} 
                unit="mmHg" 
                trend={activePatient.vitals_trend?.BP}
                history={activeHistory?.BP}
                isCritical={activePatient.vitals.systolic_bp < 90 || activePatient.vitals.systolic_bp > 180}
              />
              <VitalLine 
                label="Oxygen Sat." 
                value={activePatient.vitals.spo2} 
                unit="%" 
                trend={activePatient.vitals_trend?.SpO2}
                history={activeHistory?.SpO2}
                isCritical={activePatient.vitals.spo2 < 92}
              />
              <VitalLine 
                label="Glasgow Coma" 
                value={activePatient.vitals.gcs} 
                unit="/15" 
                trend={activePatient.vitals_trend?.GCS}
                history={activeHistory?.GCS}
                isCritical={activePatient.vitals.gcs < 13}
              />
            </div>

            <div className="mt-8 border-t border-border pt-4 flex-1">
              <span className="text-[10px] uppercase opacity-50 block mb-2">Pending Diagnostics</span>
              <div className="flex flex-wrap gap-2">
                {activePatient.pending_labs.length > 0 ? (
                  activePatient.pending_labs.map((l: any, i: number) => (
                    <span key={i} className="px-2 py-1 bg-neon-orange/10 border border-neon-orange/40 text-[10px] text-neon-orange animate-pulse">
                      WAITING_{l.toUpperCase()}
                    </span>
                  ))
                ) : (
                  <span className="text-[10px] opacity-30 italic">No pending labs.</span>
                )}
              </div>
            </div>

            <div className="mt-4 p-3 bg-crimson/5 border border-crimson/20 rounded">
              <div className="flex items-center gap-2 text-crimson mb-2">
                <AlertTriangle size={14} />
                <span className="text-[10px] uppercase font-bold">Clinical Priority Advisory</span>
              </div>
              <p className="text-[11px] opacity-80 leading-relaxed">
                {activePatient.vitals.systolic_bp < 100 ? "WARNING: Potential neuro-deterioration or shock pattern detected. Immediate stabilization required." : "Baseline established. Monitor for acute trends."}
              </p>
            </div>
          </div>
        ) : (
          <div className="h-full flex flex-col items-center justify-center opacity-30">
            <Stethoscope size={48} className="mb-4" />
            <p className="italic text-sm">Target patient not selected.</p>
          </div>
        )}
      </Panel>

      {/* Right Panel: Neural Trace */}
      <Panel title="Neural Trace / ReAct Reasoning" icon={TerminalIcon}>
        <div ref={scrollRef} className="p-4 font-mono text-xs leading-relaxed text-cyan overflow-y-auto h-full whitespace-pre-wrap">
          {thoughts.length > 0 ? (
            thoughts.join("").split("<thought>").map((chunk, i) => {
              if (i === 0) return chunk;
              const [thought, rest] = chunk.split("</thought>");
              return (
                <div key={i}>
                  <div className="bg-cyan/10 border-l-2 border-cyan p-2 my-2 text-cyan">
                    <span className="block text-[8px] uppercase opacity-50 mb-1">Inherent Reasoning Process:</span>
                    {thought}
                  </div>
                  {rest}
                </div>
              );
            })
          ) : (
            <div className="h-full flex items-center justify-center opacity-20">
              <span className="animate-pulse">AWAITING_REASONING_UPLINK...</span>
            </div>
          )}
        </div>
      </Panel>

      {/* Bottom Control Bar */}
      <div className="col-span-2 border-2 border-border bg-glass p-4 grid grid-cols-[1fr_200px] gap-6">
        <div className="flex items-center gap-4">
          <div className="flex flex-col gap-1 w-full max-w-sm">
            <span className="text-[10px] opacity-50 uppercase">Task Manifest Select</span>
            <div className="flex gap-2">
              <select 
                id="task-manifest" 
                className="flex-1 bg-black border border-border p-2 text-xs text-cyan focus:outline-none focus:border-cyan"
                onChange={(e) => reset(e.target.value)}
              >
                <option value="task_stemi_code">TASK: STEMI_CODE_ALPHA</option>
                <option value="task_chest_pain_workup">TASK: CHEST_PAIN_OMEGA</option>
                <option value="task_mci_surge">TASK: MCI_SURGE_SIGMA</option>
              </select>
              <button 
                onClick={() => reset((document.getElementById('task-manifest') as HTMLSelectElement).value)}
                className="px-4 bg-terminal-green text-black font-bold text-xs flex items-center gap-2 hover:bg-white transition-colors"
                title="Initialize simulation environment"
              >
                <RotateCcw size={14} />
                INIT
              </button>
            </div>
          </div>

          <div className="h-full w-px bg-border/50 mx-2" />

          <div className="flex-1 flex flex-col justify-center">
            <div className="flex justify-between text-[10px] uppercase mb-1">
              <span>Simulation Progress</span>
              <span className="text-terminal-green">{state?.step_number || 0} / {state?.max_steps || 15}</span>
            </div>
            <div className="h-2 w-full bg-border/30 rounded-full overflow-hidden">
              <motion.div 
                className="h-full bg-terminal-green" 
                initial={{ width: 0 }}
                animate={{ width: `${((state?.step_number || 0) / (state?.max_steps || 15)) * 100}%` }}
              />
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button 
            className="flex-1 h-full bg-cyan text-black font-black text-xs flex flex-col items-center justify-center hover:bg-white transition-colors group relative overflow-hidden disabled:opacity-50"
            disabled={!connected}
            onClick={async () => {
              addLog("System", "Manual Agent Trigger engaged. Summoning Reasoner...");
              try {
                const taskId = state?.task_id || (document.getElementById('task-manifest') as HTMLSelectElement).value;
                await fetch("/run_agent", {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ task_id: taskId })
                });
                addLog("System", "Inference sub-routine deployed successfully.");
              } catch (e) {
                addLog("Error", "Failed to deploy inference sub-routine.");
              }
            }}
          >
            <Play size={20} className="mb-1 group-hover:scale-125 transition-transform" />
            <span>RUN_AGENT</span>
            <div className="absolute inset-0 bg-white/20 translate-y-full group-active:translate-y-0 transition-transform" />
          </button>
        </div>
      </div>

      {/* Terminal Log */}
      <Panel title="Event Log / System Interrupts" icon={TerminalIcon} className="row-span-1">
        <div className="p-2 font-mono text-[10px] flex flex-col gap-1 overflow-y-auto h-full">
          {logs.map((log, i) => (
            <div key={i} className={`flex gap-2 ${log.type === 'error' ? 'text-crimson' : 'text-terminal-green/70'}`}>
              <span className="opacity-40">[{log.time}]</span>
              <span>{log.msg}</span>
            </div>
          ))}
          {logs.length === 0 && (
            <div className="h-full flex items-center justify-center opacity-10">NO_INTERRUPTS_RECORDED</div>
          )}
        </div>
      </Panel>
    </main>
  );
}
