import type { RecommendedRole } from "../types/interview";

interface RoleCardProps {
  role: RecommendedRole;
  index: number;
  selected: boolean;
  onSelect: (role: RecommendedRole) => void;
}

export function RoleCard({ role, index, selected, onSelect }: RoleCardProps) {
  return (
    <button className={`role-card ${selected ? "selected" : ""}`} onClick={() => onSelect(role)}>
      <div className="role-rank">{index + 1}</div>
      <div className="role-content">
        <h4>{role.title}</h4>
        <p>{role.match_score.toFixed(2)}% match</p>
        <small>Matched: {role.matched_skills.slice(0, 4).join(", ") || "-"}</small>
      </div>
    </button>
  );
}
