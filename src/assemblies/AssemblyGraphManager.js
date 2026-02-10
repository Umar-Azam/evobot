export class AssemblyGraphManager {
  constructor(limbCount, directedConstraintIndex, joinDistance) {
    this.limbCount = limbCount;
    this.directedConstraintIndex = directedConstraintIndex;
    this.joinDistance = joinDistance;
    this.parent = new Array(limbCount).fill(-1);
    this.children = Array.from({ length: limbCount }, () => new Set());
  }

  hasEdge(parent, child) {
    return this.parent[child] === parent;
  }

  edgeCount() {
    let count = 0;
    for (let i = 0; i < this.limbCount; i++) {
      if (this.parent[i] >= 0) {
        count += 1;
      }
    }
    return count;
  }

  isInSameComponent(a, b) {
    const rootA = this.findRoot(a);
    const rootB = this.findRoot(b);
    return rootA === rootB;
  }

  findRoot(node) {
    let cur = node;
    let guard = 0;
    while (this.parent[cur] >= 0 && guard < this.limbCount + 1) {
      cur = this.parent[cur];
      guard += 1;
    }
    return cur;
  }

  canAttach(parent, child, parentPos, childPos) {
    if (parent === child) {
      return false;
    }
    if (this.parent[child] >= 0) {
      return false;
    }
    if (this.children[parent].has(child)) {
      return false;
    }
    if (this.isInSameComponent(parent, child)) {
      return false;
    }
    const dx = parentPos[0] - childPos[0];
    const dy = parentPos[1] - childPos[1];
    const dz = parentPos[2] - childPos[2];
    const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
    return distance <= this.joinDistance;
  }

  attach(parent, child, data) {
    const key = `${parent}->${child}`;
    if (!(key in this.directedConstraintIndex)) {
      return false;
    }
    const eqId = this.directedConstraintIndex[key];
    this.parent[child] = parent;
    this.children[parent].add(child);
    data.eq_active[eqId] = 1;
    return true;
  }

  detach(child, data) {
    const parent = this.parent[child];
    if (parent < 0) {
      return false;
    }
    const key = `${parent}->${child}`;
    if (!(key in this.directedConstraintIndex)) {
      return false;
    }
    const eqId = this.directedConstraintIndex[key];
    data.eq_active[eqId] = 0;
    this.parent[child] = -1;
    this.children[parent].delete(child);
    return true;
  }

  connectedComponents() {
    const visited = new Array(this.limbCount).fill(false);
    const components = [];
    for (let i = 0; i < this.limbCount; i++) {
      if (visited[i]) {
        continue;
      }
      const stack = [i];
      const component = [];
      visited[i] = true;
      while (stack.length > 0) {
        const cur = stack.pop();
        component.push(cur);
        const p = this.parent[cur];
        if (p >= 0 && !visited[p]) {
          visited[p] = true;
          stack.push(p);
        }
        for (const c of this.children[cur]) {
          if (!visited[c]) {
            visited[c] = true;
            stack.push(c);
          }
        }
      }
      components.push(component);
    }
    return components;
  }

  topologicalPostOrder() {
    const visited = new Array(this.limbCount).fill(false);
    const order = [];

    const dfs = (node) => {
      visited[node] = true;
      for (const child of this.children[node]) {
        if (!visited[child]) {
          dfs(child);
        }
      }
      order.push(node);
    };

    for (let i = 0; i < this.limbCount; i++) {
      if (this.parent[i] === -1 && !visited[i]) {
        dfs(i);
      }
    }
    for (let i = 0; i < this.limbCount; i++) {
      if (!visited[i]) {
        dfs(i);
      }
    }
    return order;
  }
}

