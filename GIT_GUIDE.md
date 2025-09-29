# Git 最佳实践指南

## 🔧 推荐的Git配置

```bash
# 设置用户信息
git config --global user.name "你的姓名"
git config --global user.email "your.email@example.com"

# 设置默认编辑器
git config --global core.editor "code --wait"

# 设置默认分支名
git config --global init.defaultBranch main

# 启用颜色输出
git config --global color.ui auto
```

## 🚨 问题修复流程

### 1. 发现问题后立即处理

```bash
# 查看当前状态
git status

# 查看最近提交
git log --oneline -5

# 查看具体修改
git show HEAD
```

### 2. 选择合适的修复方法

#### 情况A: 问题刚发现，还没推送到远程
```bash
# 撤销最后一次提交，保留修改
git reset --soft HEAD~1

# 修复问题后重新提交
git add .
git commit -m "fix: 修复问题并重新提交"
```

#### 情况B: 已经推送到远程仓库
```bash
# 修复问题
# 提交修复
git add .
git commit -m "fix: 修复版本X.X.X中的XXX问题"
git push origin main
```

#### 情况C: 需要完全撤销某个提交
```bash
# 使用revert创建反向提交
git revert <有问题的commit-hash>
git push origin main
```

### 3. 紧急情况处理

如果问题严重影响生产环境：

```bash
# 创建hotfix分支
git checkout -b hotfix/critical-bug

# 修复问题
git add .
git commit -m "hotfix: 修复严重bug"

# 推送hotfix分支
git push origin hotfix/critical-bug

# 合并到main分支
git checkout main
git merge hotfix/critical-bug
git push origin main
```

## 📋 预防措施

1. **代码审查**：重要修改前请同事review
2. **测试**：本地充分测试后再推送
3. **分支策略**：使用feature分支开发新功能
4. **小步提交**：频繁小量提交，便于定位问题
5. **备份**：重要节点创建tag标记

## 🏷️ 版本标记

```bash
# 创建版本标签
git tag -a v1.0.0 -m "版本1.0.0发布"

# 推送标签
git push origin v1.0.0

# 回滚到特定版本
git checkout v1.0.0
```
