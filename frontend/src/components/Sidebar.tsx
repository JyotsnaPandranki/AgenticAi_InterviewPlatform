import { NavLink } from 'react-router-dom';

const items = [
  ['/', 'Home'],
  ['/upload', 'Upload Resume'],
  ['/processing', 'Processing'],
  ['/roles', 'Recommended Roles'],
  ['/init', 'Interview Init'],
  ['/live', 'Live Interview'],
  ['/summary', 'Interview Summary'],
];

export function Sidebar() {
  return (
    <aside className="hidden lg:block w-64 border-r border-white/5 p-4">
      <div className="space-y-2">
        {items.map(([to, label]) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `block rounded-xl px-3 py-2 text-sm ${isActive ? 'bg-white/10 text-on-surface' : 'text-on-surface-variant hover:bg-white/5'}`
            }
          >
            {label}
          </NavLink>
        ))}
      </div>
    </aside>
  );
}
