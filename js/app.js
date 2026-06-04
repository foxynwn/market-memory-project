/* ========================================
   TODO APP - MAIN LOGIC
   ======================================== */

class TodoApp {
    constructor() {
        this.todos = [];
        this.currentFilter = 'all';
        this.searchQuery = '';
        this.init();
    }

    init() {
        this.loadTodos();
        this.cacheDOM();
        this.bindEvents();
        this.render();
    }

    /* ========================================
       DOM CACHING
       ======================================== */

    cacheDOM() {
        // Input
        this.todoInput = document.querySelector('.todo-input');
        this.addBtn = document.querySelector('.add-btn');
        
        // Lists & Containers
        this.todosList = document.querySelector('.todos-list');
        this.todosContainer = document.querySelector('.todos-container');
        
        // Stats
        this.totalStat = document.querySelector('[data-stat="total"]');
        this.completedStat = document.querySelector('[data-stat="completed"]');
        this.pendingStat = document.querySelector('[data-stat="pending"]');
        
        // Filters
        this.filterBtns = document.querySelectorAll('.filter-btn');
        this.searchInput = document.querySelector('.search-input');
        
        // Actions
        this.clearBtn = document.querySelector('[data-action="clear"]');
        this.markAllBtn = document.querySelector('[data-action="markAll"]');
        this.exportBtn = document.querySelector('[data-action="export"]');
        
        // Modal
        this.editModal = document.querySelector('.modal');
        this.modalInput = document.querySelector('.modal-input');
        this.btnSave = document.querySelector('.btn-save');
        this.btnCancel = document.querySelector('.btn-cancel');
        this.modalClose = document.querySelector('.modal-close');
    }

    bindEvents() {
        // Input
        this.addBtn.addEventListener('click', () => this.addTodo());
        this.todoInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.addTodo();
        });
        
        // Filters
        this.filterBtns.forEach(btn => {
            btn.addEventListener('click', (e) => this.setFilter(e.target.closest('.filter-btn').dataset.filter));
        });
        
        // Search
        this.searchInput.addEventListener('input', (e) => {
            this.searchQuery = e.target.value.toLowerCase();
            this.render();
        });
        
        // Actions
        if (this.clearBtn) {
            this.clearBtn.addEventListener('click', () => this.clearCompleted());
        }
        if (this.markAllBtn) {
            this.markAllBtn.addEventListener('click', () => this.markAll());
        }
        if (this.exportBtn) {
            this.exportBtn.addEventListener('click', () => this.exportTodos());
        }
        
        // Modal
        if (this.btnSave) {
            this.btnSave.addEventListener('click', () => this.saveEdit());
        }
        if (this.btnCancel) {
            this.btnCancel.addEventListener('click', () => this.closeModal());
        }
        if (this.modalClose) {
            this.modalClose.addEventListener('click', () => this.closeModal());
        }
        
        // Close modal on outside click
        if (this.editModal) {
            this.editModal.addEventListener('click', (e) => {
                if (e.target === this.editModal) this.closeModal();
            });
        }
    }

    /* ========================================
       TODO MANAGEMENT
       ======================================== */

    addTodo() {
        const text = this.todoInput.value.trim();
        
        if (!text) {
            this.showToast('Please enter a todo', 'warning');
            return;
        }
        
        if (text.length > 500) {
            this.showToast('Todo text is too long (max 500 characters)', 'warning');
            return;
        }
        
        const todo = {
            id: Date.now(),
            text: text,
            completed: false,
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString()
        };
        
        this.todos.unshift(todo);
        this.saveTodos();
        this.todoInput.value = '';
        this.todoInput.focus();
        this.render();
        this.showToast('Todo added successfully', 'success');
    }

    deleteTodo(id) {
        const index = this.todos.findIndex(todo => todo.id === id);
        if (index > -1) {
            const deletedTodo = this.todos[index];
            this.todos.splice(index, 1);
            this.saveTodos();
            this.render();
            this.showToast(`"${deletedTodo.text}" deleted`, 'success');
        }
    }

    toggleTodo(id) {
        const todo = this.todos.find(t => t.id === id);
        if (todo) {
            todo.completed = !todo.completed;
            todo.updatedAt = new Date().toISOString();
            this.saveTodos();
            this.render();
        }
    }

    editTodo(id) {
        const todo = this.todos.find(t => t.id === id);
        if (!todo) return;
        
        this.currentEditId = id;
        this.modalInput.value = todo.text;
        this.editModal.classList.add('active');
        this.modalInput.focus();
        this.modalInput.select();
    }

    saveEdit() {
        const todo = this.todos.find(t => t.id === this.currentEditId);
        if (!todo) return;
        
        const newText = this.modalInput.value.trim();
        
        if (!newText) {
            this.showToast('Todo text cannot be empty', 'warning');
            return;
        }
        
        if (newText.length > 500) {
            this.showToast('Todo text is too long (max 500 characters)', 'warning');
            return;
        }
        
        if (newText !== todo.text) {
            todo.text = newText;
            todo.updatedAt = new Date().toISOString();
            this.saveTodos();
            this.render();
            this.showToast('Todo updated successfully', 'success');
        }
        
        this.closeModal();
    }

    closeModal() {
        this.editModal.classList.remove('active');
        this.currentEditId = null;
    }

    /* ========================================
       FILTERING & SEARCH
       ======================================== */

    setFilter(filter) {
        this.currentFilter = filter;
        
        this.filterBtns.forEach(btn => {
            btn.classList.remove('active');
            if (btn.dataset.filter === filter) {
                btn.classList.add('active');
            }
        });
        
        this.render();
    }

    getFilteredTodos() {
        let filtered = this.todos;
        
        // Apply filter
        if (this.currentFilter === 'active') {
            filtered = filtered.filter(todo => !todo.completed);
        } else if (this.currentFilter === 'completed') {
            filtered = filtered.filter(todo => todo.completed);
        }
        
        // Apply search
        if (this.searchQuery) {
            filtered = filtered.filter(todo => 
                todo.text.toLowerCase().includes(this.searchQuery)
            );
        }
        
        return filtered;
    }

    /* ========================================
       BULK ACTIONS
       ======================================== */

    clearCompleted() {
        const completed = this.todos.filter(t => t.completed);
        
        if (completed.length === 0) {
            this.showToast('No completed todos to clear', 'info');
            return;
        }
        
        if (confirm(`Delete ${completed.length} completed todo(s)?`)) {
            this.todos = this.todos.filter(t => !t.completed);
            this.saveTodos();
            this.render();
            this.showToast(`${completed.length} completed todo(s) deleted`, 'success');
        }
    }

    markAll() {
        const hasIncomplete = this.todos.some(t => !t.completed);
        const newState = hasIncomplete;
        
        this.todos.forEach(todo => {
            if (todo.completed !== newState) {
                todo.completed = newState;
                todo.updatedAt = new Date().toISOString();
            }
        });
        
        this.saveTodos();
        this.render();
        
        const action = newState ? 'marked as complete' : 'marked as incomplete';
        this.showToast(`All todos ${action}`, 'success');
    }

    exportTodos() {
        if (this.todos.length === 0) {
            this.showToast('No todos to export', 'warning');
            return;
        }
        
        const dataStr = JSON.stringify(this.todos, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `todos_${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        
        this.showToast('Todos exported successfully', 'success');
    }

    /* ========================================
       STORAGE
       ======================================== */

    saveTodos() {
        try {
            localStorage.setItem('todos', JSON.stringify(this.todos));
        } catch (e) {
            console.error('Failed to save todos:', e);
            this.showToast('Failed to save todos', 'error');
        }
    }

    loadTodos() {
        try {
            const stored = localStorage.getItem('todos');
            this.todos = stored ? JSON.parse(stored) : [];
            
            // Validation
            if (!Array.isArray(this.todos)) {
                this.todos = [];
            }
        } catch (e) {
            console.error('Failed to load todos:', e);
            this.todos = [];
        }
    }

    /* ========================================
       RENDERING
       ======================================== */

    render() {
        this.updateStats();
        this.renderTodos();
    }

    updateStats() {
        const total = this.todos.length;
        const completed = this.todos.filter(t => t.completed).length;
        const pending = total - completed;
        
        if (this.totalStat) this.totalStat.textContent = total;
        if (this.completedStat) this.completedStat.textContent = completed;
        if (this.pendingStat) this.pendingStat.textContent = pending;
    }

    renderTodos() {
        const filtered = this.getFilteredTodos();
        
        if (filtered.length === 0) {
            this.todosList.innerHTML = '';
            this.todosContainer.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">📭</div>
                    <div class="empty-text">No todos here</div>
                    <div class="empty-hint">
                        ${this.searchQuery ? 'Try a different search' : 'Add one to get started'}
                    </div>
                </div>
            `;
            return;
        }
        
        this.todosList.innerHTML = filtered.map(todo => this.renderTodoItem(todo)).join('');
        
        // Re-bind item events
        this.bindTodoItemEvents();
    }

    renderTodoItem(todo) {
        const createdDate = new Date(todo.createdAt).toLocaleDateString();
        
        return `
            <div class="todo-item ${todo.completed ? 'completed' : ''}">
                <input 
                    type="checkbox" 
                    class="todo-checkbox" 
                    data-id="${todo.id}"
                    ${todo.completed ? 'checked' : ''}
                />
                <div class="todo-content">
                    <div class="todo-text">${this.escapeHtml(todo.text)}</div>
                    <div class="todo-meta">Created: ${createdDate}</div>
                </div>
                <div class="todo-actions">
                    <button class="todo-action-btn edit" data-id="${todo.id}" title="Edit">✎</button>
                    <button class="todo-action-btn delete" data-id="${todo.id}" title="Delete">🗑</button>
                </div>
            </div>
        `;
    }

    bindTodoItemEvents() {
        // Checkboxes
        document.querySelectorAll('.todo-checkbox').forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                this.toggleTodo(parseInt(e.target.dataset.id));
            });
        });
        
        // Edit buttons
        document.querySelectorAll('.todo-action-btn.edit').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.editTodo(parseInt(e.target.dataset.id));
            });
        });
        
        // Delete buttons
        document.querySelectorAll('.todo-action-btn.delete').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const id = parseInt(e.target.dataset.id);
                const todo = this.todos.find(t => t.id === id);
                if (confirm(`Delete "${todo.text}"?`)) {
                    this.deleteTodo(id);
                }
            });
        });
    }

    /* ========================================
       UTILITIES
       ======================================== */

    escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, m => map[m]);
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        document.body.appendChild(toast);
        
        // Trigger animation
        setTimeout(() => toast.classList.add('show'), 10);
        
        // Remove after 3 seconds
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => document.body.removeChild(toast), 300);
        }, 3000);
    }
}

/* ========================================
   INITIALIZATION
   ======================================== */

document.addEventListener('DOMContentLoaded', () => {
    window.app = new TodoApp();
});