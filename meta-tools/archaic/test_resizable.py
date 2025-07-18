#!/usr/bin/env python3
"""
Test script to verify that the resizable window functionality works correctly.
This script creates a simple version to test the window resizing.
"""

import pygame
import sys
import os

def test_resizable_window():
    """Test the resizable window functionality."""
    pygame.init()
    
    # Initial window size
    screen_width = 800
    screen_height = 600
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
    pygame.display.set_caption("Test Resizable Window")
    
    font = pygame.font.SysFont(None, 30)
    clock = pygame.time.Clock()
    
    # Test variables
    buttons = []
    test_dirs = ["Documents", "Pictures", "Videos"]
    
    def recalculate_layout():
        nonlocal buttons, font
        
        # Adaptive font size
        font_size = max(20, min(40, screen_width // 40))
        font = pygame.font.SysFont(None, font_size)
        
        # Recalculate button layout
        button_height = max(40, min(80, screen_height // 10))
        button_width = screen_width // len(test_dirs)
        
        buttons = []
        for idx, dirname in enumerate(test_dirs):
            rect = pygame.Rect(idx * button_width, screen_height - button_height, button_width, button_height)
            buttons.append((rect, dirname))
    
    # Initial layout
    recalculate_layout()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                # Enforce minimum size
                min_width = max(600, len(test_dirs) * 100)
                min_height = 400
                screen_width = max(min_width, event.w)
                screen_height = max(min_height, event.h)
                
                screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
                recalculate_layout()
                pygame.display.set_caption(f"Test Resizable Window ({screen_width}x{screen_height})")
        
        # Draw
        screen.fill((0, 0, 0))
        
        # Draw test text
        text = font.render(f"Window Size: {screen_width}x{screen_height}", True, (255, 255, 255))
        screen.blit(text, (10, 10))
        
        # Draw buttons
        for rect, label in buttons:
            pygame.draw.rect(screen, (0, 255, 0), rect)
            
            # Truncate label if needed
            max_label_width = rect.width - 10
            text = font.render(label, True, (0, 0, 0))
            if text.get_width() > max_label_width:
                truncated_label = label
                while font.render(truncated_label + "...", True, (0, 0, 0)).get_width() > max_label_width and len(truncated_label) > 3:
                    truncated_label = truncated_label[:-1]
                truncated_label += "..."
                text = font.render(truncated_label, True, (0, 0, 0))
            
            text_rect = text.get_rect(center=rect.center)
            screen.blit(text, text_rect)
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    print("Test completed successfully!")

if __name__ == "__main__":
    test_resizable_window()
