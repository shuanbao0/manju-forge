"use client";

import { motion } from "framer-motion";
import { Calendar, Trash2, Play } from "lucide-react";
import { useRouter } from "next/navigation";
import { Project } from "@/store/projectStore";
import { useTranslation } from "@/i18n";
import { confirmDialog } from "@/components/common/dialogs";

interface ProjectCardProps {
    project: Project;
    onDelete: (id: string) => void;
}

export default function ProjectCard({ project, onDelete }: ProjectCardProps) {
    const router = useRouter();
    const { t } = useTranslation();

    const handleOpen = () => {
        // Flow B projects use the dedicated Remotion MG editor.
        if (project.generation_engine === "remotion_mg") {
            window.location.hash = `#/mg/${project.id}`;
        } else {
            window.location.hash = `#/project/${project.id}`;
        }
    };

    const handleDelete = async (e: React.MouseEvent) => {
        e.stopPropagation();
        const ok = await confirmDialog({
            title: t("project.deleteTitle", undefined, "删除项目"),
            message: t("project.confirmDelete", { title: project.title }),
            variant: "danger",
            confirmLabel: t("common.delete", undefined, "删除"),
        });
        if (ok) onDelete(project.id);
    };

    const statusColors = {
        pending: "bg-gray-500/20 text-gray-400",
        processing: "bg-yellow-500/20 text-yellow-400",
        completed: "bg-green-500/20 text-green-400",
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            whileHover={{ scale: 1.02 }}
            className="glass-panel p-6 rounded-xl cursor-pointer group relative border-l-2 border-l-gray-600"
            onClick={handleOpen}
        >
            <div className="flex items-start justify-between mb-4">
                <div className="flex-1">
                    <h3 className="text-lg font-display font-bold text-white mb-2">
                        {project.title}
                    </h3>
                    <div className="flex items-center gap-2 text-xs text-gray-400">
                        <Calendar size={12} />
                        <span>{new Date(project.createdAt).toLocaleDateString('zh-CN')}</span>
                    </div>
                </div>

                <div className="flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                        onClick={handleDelete}
                        className="p-2 rounded-lg bg-red-500/10 hover:bg-red-500/20 text-red-400 transition-colors"
                    >
                        <Trash2 size={16} />
                    </button>
                </div>
            </div>

            <div className="flex items-center gap-3 text-xs text-gray-400 mb-4">
                <span>{t("series.charactersHeading")} <span className="text-white font-medium">{project.characters?.length || 0}</span></span>
                <span className="text-gray-600">·</span>
                <span>{t("series.scenesHeading")} <span className="text-white font-medium">{project.scenes?.length || 0}</span></span>
                <span className="text-gray-600">·</span>
                <span>{t("modules.storyboard.frameLabel")} <span className="text-white font-medium">{project.frames?.length || 0}</span></span>
            </div>

            <div className="flex items-center justify-between">
                <span className={`text-xs px-2 py-1 rounded ${statusColors[project.status as keyof typeof statusColors] || statusColors.pending}`}>
                    {project.status || t("projectCard.statusPending", undefined, "待开始")}
                </span>

                <div className="flex items-center gap-1 text-primary text-xs font-medium">
                    <Play size={14} />
                    <span>{t("projectCard.openProject", undefined, "打开项目")}</span>
                </div>
            </div>
        </motion.div>
    );
}
