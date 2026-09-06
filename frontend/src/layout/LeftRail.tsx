import { NavLink } from 'react-router-dom'
import type { ReactNode } from 'react'

// Inline icons rather than an icon package: adding a dependency is a hard
// stop, and six glyphs do not justify one.
function Icon({ children }: { children: ReactNode }) {
  return (
    <svg
      viewBox="0 0 20 20"
      className="size-5 shrink-0"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {children}
    </svg>
  )
}

const NAV = [
  {
    to: '/map',
    label: 'Live map',
    icon: (
      <Icon>
        <path d="M2.5 5.5 7.5 3l5 2.5L17.5 3v11.5L12.5 17l-5-2.5L2.5 17z" />
        <path d="M7.5 3v11.5M12.5 5.5V17" />
      </Icon>
    ),
  },
  {
    to: '/search',
    label: 'Search',
    icon: (
      <Icon>
        <circle cx="8.5" cy="8.5" r="5.5" />
        <path d="m12.5 12.5 4.5 4.5" />
      </Icon>
    ),
  },
  {
    to: '/journey',
    label: 'Journey',
    icon: (
      <Icon>
        <circle cx="4.5" cy="15.5" r="2" />
        <circle cx="15.5" cy="4.5" r="2" />
        <path d="M6.5 15.5h4a3 3 0 0 0 3-3v-4" strokeDasharray="2 2" />
      </Icon>
    ),
  },
  {
    to: '/alerts',
    label: 'Alerts',
    icon: (
      <Icon>
        <path d="M10 2.5 18 16.5H2z" />
        <path d="M10 8v3.5M10 14h.01" />
      </Icon>
    ),
  },
  {
    to: '/cameras',
    label: 'Cameras',
    icon: (
      <Icon>
        <rect x="2.5" y="5.5" width="11" height="9" />
        <path d="m13.5 10 4-2.5v5z" />
      </Icon>
    ),
  },
  {
    to: '/status',
    label: 'System status',
    icon: (
      <Icon>
        <path d="M2.5 10h3l2-5 3 10 2.5-5h4.5" />
      </Icon>
    ),
  },
]

// Icon AND visible text label for every route. No hamburger, no icon-only
// rail, no hover-only affordance: a projector audience never sees a hover
// state, and a menu that has to be opened is a menu they never see at all.
export function LeftRail() {
  return (
    <nav
      aria-label="Main"
      className="w-[184px] shrink-0 border-r border-rule bg-panel"
    >
      <ul className="flex flex-col gap-0.5 p-2">
        {NAV.map((item) => (
          <li key={item.to}>
            <NavLink
              to={item.to}
              className={({ isActive }) =>
                [
                  'flex items-center gap-2.5 rounded-[4px] px-2.5 py-2 text-[0.95rem]',
                  isActive
                    ? 'bg-sunken font-medium text-ink'
                    : 'text-ink-2 hover:bg-sunken hover:text-ink',
                ].join(' ')
              }
            >
              {item.icon}
              <span>{item.label}</span>
            </NavLink>
          </li>
        ))}
      </ul>
    </nav>
  )
}
