const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const html = fs.readFileSync(path.join(__dirname, "..", "index.html"), "utf8");
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(match => match[1]);

function engine() {
  const context = vm.createContext({ document: { getElementById: () => ({}) } });
  vm.runInContext(scripts[1].split("    function navigation()")[0], context);
  return source => vm.runInContext(source, context);
}

test("the self-contained guide has valid scripts and all learning mechanisms", () => {
  assert.equal(scripts.length, 2);
  scripts.forEach(script => new vm.Script(script));
  const evaluate = engine();
  assert.equal(evaluate("LESSONS.length"), 8);
  assert.equal(evaluate("LESSONS.every(l=>l.steps.length===3 && l.quiz.length===4 && l.sources.length>0)"), true);
  assert.equal(evaluate("LESSONS.every(l=>{const d=demo(l.id);return d.plot.length>100 && d.controls.length>100 && d.insight.length>40;})"), true);
  assert.doesNotMatch(html, /<script[^>]+src=|<img[^>]+src=|localStorage|sessionStorage|\bfetch\s*\(/);
});

test("least squares minimizes training error and reacts to an outlier", () => {
  const evaluate = engine();
  const result = JSON.parse(evaluate(`JSON.stringify((()=>{
    const rows=linearData(),fit=linearFit(rows);
    const error=mse(rows,x=>fit.slope*x+fit.intercept);
    const alternatives=[-0.2,0.2].flatMap(ds=>[-0.2,0.2].map(di=>mse(rows,x=>(fit.slope+ds)*x+fit.intercept+di)));
    state.outlier=true;
    const changed=linearFit(linearData());
    return{error,alternatives,slope:fit.slope,changedSlope:changed.slope};
  })())`));
  assert.ok(result.alternatives.every(error => error >= result.error));
  assert.notEqual(result.slope, result.changedSlope);
  assert.throws(() => evaluate("linearFit([{x:1,y:1},{x:1,y:2}])"), /constant inputs/);
});

test("tree depth lowers training loss but can worsen held-out validation loss", () => {
  const evaluate = engine();
  const losses = JSON.parse(evaluate(`JSON.stringify(Array.from({length:8},(_,depth)=>{
    const tree=fitTree(TRAIN,depth),predict=x=>predictTree(tree,x);
    return{depth,leaves:leafCount(tree),train:mse(TRAIN,predict),validation:mse(VALID,predict)};
  }))`));
  assert.equal(losses[0].leaves, 1);
  for (let i = 1; i < losses.length; i++) {
    assert.ok(losses[i].train <= losses[i - 1].train + 1e-12);
    assert.ok(losses[i].leaves >= losses[i - 1].leaves);
  }
  assert.ok(losses[6].validation > losses[3].validation);
});

test("bootstrap ensembles are deterministic and aggregate independent views", () => {
  const evaluate = engine();
  const first = evaluate("JSON.stringify(forest(12,3))");
  assert.equal(first, evaluate("JSON.stringify(forest(12,3))"));
  const predictions = JSON.parse(evaluate("JSON.stringify(forest(12,3).map(t=>predictTree(t,.65)))"));
  assert.ok(new Set(predictions).size > 1);
  const average = predictions.reduce((sum, value) => sum + value, 0) / predictions.length;
  assert.ok(average >= Math.min(...predictions) && average <= Math.max(...predictions));
  assert.equal(evaluate("forest(1,3).length"), 1);
});

test("squared-error residual boosting respects zero rate and reduces training loss", () => {
  const evaluate = engine();
  const zero = JSON.parse(evaluate("JSON.stringify(boosted(20,0).history)"));
  assert.ok(zero.every(value => value === zero[0]));
  for (const rate of [0.05, 0.3, 0.75, 1]) {
    const loss = JSON.parse(evaluate(`JSON.stringify(boosted(20,${rate}).history)`));
    assert.equal(loss.length, 21);
    for (let i = 1; i < loss.length; i++) assert.ok(loss[i] <= loss[i - 1] + 1e-12);
    assert.ok(loss.at(-1) < loss[0]);
  }
});

test("ordered categorical encoding excludes each row's own target", () => {
  const evaluate = engine();
  for (let index = 0; index < 8; index++) {
    const result = JSON.parse(evaluate(`JSON.stringify((()=>{
      state.catRow=${index};state.catFlip=false;const original=categoryRows();
      state.catFlip=true;const flipped=categoryRows();
      return{orderedBefore:encodeCategories(original,true)[${index}],orderedAfter:encodeCategories(flipped,true)[${index}],naiveBefore:encodeCategories(original,false)[${index}],naiveAfter:encodeCategories(flipped,false)[${index}]};
    })())`));
    assert.equal(result.orderedBefore, result.orderedAfter);
    assert.notEqual(result.naiveBefore, result.naiveAfter);
  }
  assert.equal(evaluate("encodeCategories(CAT,true)[0]"), 0.5);
});

test("PCA maximizes projected variance and obeys reconstruction identities", () => {
  const evaluate = engine();
  for (const scale of [0.25, 1, 3]) {
    const result = JSON.parse(evaluate(`JSON.stringify((()=>{
      const data=pcaData(${scale}),base=pcaInfo(data,0),best=pcaInfo(data,base.bestAngle);
      const candidates=Array.from({length:181},(_,i)=>pcaInfo(data,i-90).ratio);
      return{best,candidates};
    })())`));
    assert.ok(result.best.ratio > 0 && result.best.ratio <= 1 + 1e-12);
    assert.ok(result.candidates.every(ratio => ratio <= result.best.ratio + 1e-12));
    assert.ok(Math.abs(result.best.ratio - result.best.bestRatio) < 1e-12);
    assert.ok(Math.abs(result.best.retained + result.best.reconstruction - result.best.total) < 1e-12);
  }
  const angle1 = evaluate("pcaInfo(pcaData(1),0).bestAngle");
  const angle3 = evaluate("pcaInfo(pcaData(3),0).bestAngle");
  assert.notEqual(angle1, angle3);
});

test("model-selection advice distinguishes prediction from compression", () => {
  const evaluate = engine();
  assert.equal(evaluate("state.task='compress';modelAdvice()[0][0]"), "PCA");
  assert.equal(evaluate("state.task='class';modelAdvice()[0][0]"), "Logistic regression");
  assert.equal(evaluate("state.task='number';modelAdvice()[0][0]"), "Linear regression");
  assert.equal(evaluate("state.structure='categories';modelAdvice()[1][0]"), "CatBoost candidate");
  assert.equal(evaluate("state.structure='numeric';state.constraint='accuracy';modelAdvice()[1][0]"), "XGBoost and random-forest candidates");
});
