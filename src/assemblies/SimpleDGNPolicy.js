function deterministicWeight(i, j, scale = 1.0) {
  const x = Math.sin((i + 1) * 12.9898 + (j + 1) * 78.233) * 43758.5453;
  return ((x - Math.floor(x)) * 2.0 - 1.0) * scale;
}

function dot(a, b) {
  let out = 0.0;
  for (let i = 0; i < a.length; i++) {
    out += a[i] * b[i];
  }
  return out;
}

function sigmoid(x) {
  return 1.0 / (1.0 + Math.exp(-x));
}

export class SimpleDGNPolicy {
  constructor(obsDim, msgDim = 8, hiddenDim = 16) {
    this.obsDim = obsDim;
    this.msgDim = msgDim;
    this.hiddenDim = hiddenDim;
    this.inDim = obsDim + msgDim;

    this.w1 = Array.from({ length: hiddenDim }, (_, i) =>
      Array.from({ length: this.inDim }, (_, j) => deterministicWeight(i, j, 0.45))
    );
    this.b1 = Array.from({ length: hiddenDim }, (_, i) => deterministicWeight(i, 301, 0.15));

    this.wTorque = Array.from({ length: 3 }, (_, k) =>
      Array.from({ length: hiddenDim }, (_, j) => deterministicWeight(k + 80, j, 0.35))
    );
    this.wAttach = Array.from({ length: hiddenDim }, (_, j) => deterministicWeight(120, j, 0.4));
    this.wDetach = Array.from({ length: hiddenDim }, (_, j) => deterministicWeight(140, j, 0.4));
    this.wMsg = Array.from({ length: msgDim }, (_, i) =>
      Array.from({ length: hiddenDim }, (_, j) => deterministicWeight(i + 200, j, 0.35))
    );
  }

  forward(obsVec, childMsgVec, obsMeta) {
    const x = obsVec.concat(childMsgVec);
    const h = new Array(this.hiddenDim);
    for (let i = 0; i < this.hiddenDim; i++) {
      h[i] = Math.tanh(dot(this.w1[i], x) + this.b1[i]);
    }

    const torque = new Array(3);
    for (let k = 0; k < 3; k++) {
      torque[k] = Math.tanh(dot(this.wTorque[k], h));
    }

    // Heuristic bias terms make the untrained network operational for attach/detach.
    const attachBias =
      (obsMeta.nearestChild >= 0 ? 1.2 : -1.2) +
      (obsMeta.nearestDist < 0.35 ? 0.8 : 0.0) +
      (obsMeta.hasParent ? -0.8 : 0.4) +
      (obsMeta.childCount >= 2 ? -0.5 : 0.3);
    const detachBias =
      (obsMeta.hasParent ? -0.25 : -1.3) +
      (obsMeta.height < 0.08 ? 0.25 : -0.3) +
      (obsMeta.childCount > 0 ? -0.5 : 0.0);

    const attachProb = sigmoid(dot(this.wAttach, h) + attachBias);
    const detachProb = sigmoid(dot(this.wDetach, h) + detachBias);

    const outMsg = new Array(this.msgDim);
    for (let i = 0; i < this.msgDim; i++) {
      outMsg[i] = Math.tanh(dot(this.wMsg[i], h));
    }

    return {
      torque,
      attachProb,
      detachProb,
      message: outMsg,
    };
  }

  getWeights() {
    return {
      w1: this.w1.map((row) => row.slice()),
      b1: this.b1.slice(),
      wTorque: this.wTorque.map((row) => row.slice()),
      wAttach: this.wAttach.slice(),
      wDetach: this.wDetach.slice(),
      wMsg: this.wMsg.map((row) => row.slice()),
    };
  }

  setWeights(weights) {
    const copyMatrix = (dst, src) => {
      if (!Array.isArray(src) || src.length !== dst.length) {
        return;
      }
      for (let i = 0; i < dst.length; i++) {
        if (!Array.isArray(src[i]) || src[i].length !== dst[i].length) {
          return;
        }
      }
      for (let i = 0; i < dst.length; i++) {
        for (let j = 0; j < dst[i].length; j++) {
          const v = src[i][j];
          if (!Number.isFinite(v)) {
            return;
          }
        }
      }
      for (let i = 0; i < dst.length; i++) {
        for (let j = 0; j < dst[i].length; j++) {
          dst[i][j] = src[i][j];
        }
      }
    };

    const copyVector = (dst, src) => {
      if (!Array.isArray(src) || src.length !== dst.length) {
        return;
      }
      for (let i = 0; i < dst.length; i++) {
        if (!Number.isFinite(src[i])) {
          return;
        }
      }
      for (let i = 0; i < dst.length; i++) {
        dst[i] = src[i];
      }
    };

    if (!weights || typeof weights !== "object") {
      return;
    }
    if ("w1" in weights) {
      copyMatrix(this.w1, weights.w1);
    }
    if ("b1" in weights) {
      copyVector(this.b1, weights.b1);
    }
    if ("wTorque" in weights) {
      copyMatrix(this.wTorque, weights.wTorque);
    }
    if ("wAttach" in weights) {
      copyVector(this.wAttach, weights.wAttach);
    }
    if ("wDetach" in weights) {
      copyVector(this.wDetach, weights.wDetach);
    }
    if ("wMsg" in weights) {
      copyMatrix(this.wMsg, weights.wMsg);
    }
  }
}
