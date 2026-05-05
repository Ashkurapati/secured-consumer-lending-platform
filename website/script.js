// ── Config ──────────────────────────────────────────────────
const API_URL = "/api/predict";
let currentStep = 1;
const TOTAL_STEPS = 4;

// ── Navigation ───────────────────────────────────────────────

function goNext(fromStep) {
  if (!validateStep(fromStep)) return;
  showStep(fromStep + 1);
}

function goBack(fromStep) {
  if (fromStep === 4) {
    currentStep = 1;
    showStep(1);
    resetResults();
    return;
  }
  showStep(fromStep - 1);
}

function showStep(step) {
  for (let i = 1; i <= TOTAL_STEPS; i++) {
    const el = document.getElementById(`step-${i}`);
    if (el) el.classList.toggle("hidden", i !== step);
  }
  updateProgress(step);
  currentStep = step;
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function updateProgress(step) {
  for (let i = 1; i <= TOTAL_STEPS; i++) {
    const prog = document.getElementById(`prog-${i}`);
    if (!prog) continue;
    prog.classList.remove("active", "done");
    if (i < step) prog.classList.add("done");
    else if (i === step) prog.classList.add("active");
  }
}

// ── Guarantor toggle ─────────────────────────────────────────

function toggleGuarantor(show) {
  const fields = document.getElementById("guarantor-fields");
  if (show) fields.classList.remove("hidden");
  else fields.classList.add("hidden");
}

// ── Auto-calculate credit card utilisation ───────────────────

function autoCalcUtil() {
  const limit   = parseFloat(document.getElementById("total_bc_limit").value) || 0;
  const balance = parseFloat(document.getElementById("revol_bal").value) || 0;
  const util    = document.getElementById("revol_util");
  if (limit > 0) {
    util.value = Math.min(100, (balance / limit * 100).toFixed(1));
  } else {
    util.value = "";
  }
}

// ── Validation ───────────────────────────────────────────────

function validateStep(step) {
  const card     = document.getElementById(`step-${step}`);
  const required = card.querySelectorAll("[required]");
  let valid = true;

  required.forEach(el => {
    el.style.borderColor = "";
    if (!el.value.trim()) {
      el.style.borderColor = "#e53935";
      if (valid) el.focus();
      valid = false;
    }
  });

  if (!valid) { showToast("Please fill in all required fields."); return false; }

  // ── Step 1 specific ──────────────────────────────────────
  if (step === 1) {
    // Age check — must be 18+
    const dob = document.getElementById("dob").value;
    if (dob) {
      const today    = new Date();
      const dobDate  = new Date(dob);
      let age        = today.getFullYear() - dobDate.getFullYear();
      const m        = today.getMonth() - dobDate.getMonth();
      if (m < 0 || (m === 0 && today.getDate() < dobDate.getDate())) age--;
      if (age < 18) {
        markError("dob", "You must be at least 18 years old to apply.");
        return false;
      }
    }
  }

  // ── Step 2 specific ──────────────────────────────────────
  if (step === 2) {
    const cs = parseInt(getValue("credit_score"));
    if (cs < 300 || cs > 850) {
      markError("credit_score", "Credit score must be between 300 and 850.");
      return false;
    }

    const loan = parseFloat(getValue("loan_amnt"));
    if (loan < 200 || loan > 5000) {
      markError("loan_amnt", "Loan amount must be between $200 and $5,000.");
      return false;
    }

    // Ensure revol_util is filled (auto-calc may not have run if limit is 0)
    const util = document.getElementById("revol_util");
    if (!util.value) util.value = "0";
  }

  // ── Step 3 specific ──────────────────────────────────────
  if (step === 3) {
    const hasGuarantor = document.querySelector('input[name="has_guarantor"]:checked')?.value === "yes";
    if (hasGuarantor) {
      const borrowerPhone  = getValue("phone").replace(/\D/g, "");
      const guarantorPhone = getValue("g_phone").replace(/\D/g, "");
      if (borrowerPhone && guarantorPhone && borrowerPhone === guarantorPhone) {
        markError("g_phone", "Guarantor phone number cannot be the same as the borrower's phone number.");
        return false;
      }
    }
  }

  return true;
}

function markError(id, msg) {
  const el = document.getElementById(id);
  if (el) { el.style.borderColor = "#e53935"; el.focus(); }
  showToast(msg);
}

function showToast(msg) {
  let toast = document.getElementById("toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.id = "toast";
    toast.style.cssText = `
      position:fixed; bottom:28px; left:50%; transform:translateX(-50%);
      background:#c62828; color:#fff; padding:12px 24px;
      border-radius:8px; font-size:0.9rem; z-index:9999;
      box-shadow:0 4px 16px rgba(0,0,0,0.2); transition:opacity 0.3s;
    `;
    document.body.appendChild(toast);
  }
  toast.textContent = msg;
  toast.style.opacity = "1";
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => { toast.style.opacity = "0"; }, 3500);
}

// ── Helpers ──────────────────────────────────────────────────

function getValue(id) { const el = document.getElementById(id); return el ? el.value : ""; }
function getFloat(id) { return parseFloat(getValue(id)) || 0; }
function getInt(id)   { return parseInt(getValue(id)) || 0; }

function fmt(n)     { return "$" + n.toLocaleString("en-US", {minimumFractionDigits:2, maximumFractionDigits:2}); }
function fmtRate(n) { return n.toFixed(1) + "%"; }

// ── Submit ───────────────────────────────────────────────────

async function submitApplication() {
  if (!validateStep(3)) return;

  showStep(4);

  const hasGuarantor   = document.querySelector('input[name="has_guarantor"]:checked')?.value === "yes";
  const relationship   = hasGuarantor ? (getValue("g_relationship") || "friend") : "none";
  const creditHistYrs  = getFloat("credit_history_yrs");
  const moSinOldIl     = Math.round(creditHistYrs * 12);
  const selectedTerm   = getInt("term");

  const payload = {
    first_name:           getValue("first_name"),
    last_name:            getValue("last_name"),
    home_ownership:       getValue("home_ownership"),
    emp_length:           getInt("emp_length"),
    loan_amnt:            getFloat("loan_amnt"),
    purpose:              getValue("purpose"),
    term:                 selectedTerm,
    annual_inc:           getFloat("annual_inc"),
    credit_score:         getInt("credit_score"),
    monthly_debt:         getFloat("monthly_debt"),
    revol_util:           getFloat("revol_util"),
    revol_bal:            getFloat("revol_bal"),
    total_bc_limit:       getFloat("total_bc_limit"),
    total_acc:            getInt("total_acc"),
    avg_cur_bal:          getFloat("avg_cur_bal"),
    total_bal_ex_mort:    getFloat("total_bal_ex_mort"),
    tot_hi_cred_lim:      getFloat("tot_hi_cred_lim"),
    mo_sin_old_il_acct:   moSinOldIl,
    delinq_2yrs:          getInt("delinq_2yrs"),
    pub_rec:              getInt("pub_rec"),
    pub_rec_bankruptcies: getInt("pub_rec_bankruptcies"),
    has_guarantor:        hasGuarantor,
    guarantor_relationship: relationship,
  };

  try {
    const resp = await fetch(API_URL, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });
    if (!resp.ok) throw new Error(`Server error: ${resp.status}`);
    const data = await resp.json();
    displayResult(data, selectedTerm);
  } catch (err) {
    document.getElementById("loading").classList.add("hidden");
    showToast("Could not connect to the server. Is the API running on port 8000?");
    console.error(err);
  }
}

// ── Display Result ───────────────────────────────────────────

function displayResult(data, term) {
  document.getElementById("loading").classList.add("hidden");

  if (data.status === "rejected") {
    document.getElementById("rejection-reason").textContent = data.rejection_reason;
    document.getElementById("result-rejected").classList.remove("hidden");
    return;
  }

  document.getElementById("res-rate").textContent       = fmtRate(data.interest_rate) + " p.a.";
  document.getElementById("res-collateral").textContent = fmt(data.collateral_required);
  document.getElementById("res-monthly").textContent    = fmt(data.monthly_payment);
  document.getElementById("res-term").textContent       = term + " months";
  document.getElementById("res-total").textContent      = fmt(data.total_repayment);
  document.getElementById("res-interest").textContent   = fmt(data.total_interest);

  const tbody = document.getElementById("schedule-body");
  tbody.innerHTML = "";
  data.repayment_schedule.forEach(row => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.month}</td>
      <td>${fmt(row.payment)}</td>
      <td>${fmt(row.principal)}</td>
      <td>${fmt(row.interest)}</td>
      <td>${fmt(row.balance)}</td>
    `;
    tbody.appendChild(tr);
  });

  document.getElementById("result-approved").classList.remove("hidden");
}

function resetResults() {
  document.getElementById("loading").classList.remove("hidden");
  document.getElementById("result-rejected").classList.add("hidden");
  document.getElementById("result-approved").classList.add("hidden");
}

// ── Init ─────────────────────────────────────────────────────
showStep(1);
